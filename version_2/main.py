"""
Main experiment: LRE-GAT over-squashing analysis.

Datasets:
  ring   → RingTransfer (sintético, over-squashing garantizado)
  mnist  → MNIST Superpíxeles (visión real, ~75 nodos/grafo)
  all    → Ambos

Uso:
  python main.py --dataset ring          # Solo RingTransfer (~10 min CPU)
  python main.py --dataset mnist         # Solo MNIST (~40 min CPU, ~15 min MPS)
  python main.py --dataset all           # Ambos
  python main.py --dataset ring --quiet  # Sin logs de epoch
"""

import argparse
import json
import os
import random
import torch
import numpy as np

from datasets import make_ring_transfer, load_mnist_superpixels
from rewiring import STRATEGIES, random_rewiring, khop_rewiring, feature_similarity_rewiring
from models import GAT, GATGraphLevel
from train import train_graph_list, train_graph_level, get_device
from metrics import (
    jacobian_norm_by_distance,
    graph_level_jacobian_norm_by_distance,
    distance_bucket_counts,
    compute_fidelity,
    permutation_test,
    explanation_sparsity,
)
from visualize import (
    plot_report,
    plot_attention_heatmap,
    plot_attention_comparison,
    summarize_attention_on_artificial,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RING_CONFIG  = dict(num_nodes=20, num_classes=5, num_graphs=500)
GAT_KWARGS   = dict(hidden=32, heads=4, num_layers=3, dropout=0.5)
TRAIN_RING   = dict(epochs=200, lr=0.005, weight_decay=5e-4)

MNIST_CONFIG = dict(subset_train=10000, subset_test=2000)

GAT_MNIST = dict(
    hidden=64,
    heads=4,
    num_layers=4,       # 3 → 4: más expresividad para 10 clases
    dropout=0.3,        # solo en MLP final, no en GATConv interno
)

TRAIN_MNIST = dict(
    epochs=50,
    lr=1e-3,            # 5e-3 → 1e-3: 5e-3 oscilaba demasiado con 10 clases
    weight_decay=5e-5,  # 1e-4 → 5e-5: regularización más suave
    batch_size=128,     # 64 → 128: mejor estimación del gradiente
)

# K-hop budget: en RingTransfer queremos k grande (10 hops a recortar);
# en MNIST con ~75 nodos, k=3 ya añade muchas aristas — k=5 satura el grafo.
KHOP_K_RING  = 3
KHOP_K_MNIST = 3


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

SEED = 42

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_results(results: dict, models: dict, prefix: str):
    """Save metrics to JSON and model weights to .pt files under results/."""
    os.makedirs("results", exist_ok=True)

    # Metrics: drop non-serialisable keys (jacobian_by_dist has int keys)
    serialisable = {}
    for strategy, r in results.items():
        serialisable[strategy] = {
            k: ({str(dk): dv for dk, dv in v.items()} if isinstance(v, dict) else v)
            for k, v in r.items()
        }
    metrics_path = f"results/{prefix}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"[save] Metrics → {metrics_path}")

    for strategy, model in models.items():
        safe_name = strategy.lower().replace(" ", "_")
        model_path = f"results/{prefix}_model_{safe_name}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"[save] Model {strategy} → {model_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_strategy(name: str, raw_graphs: list, k: int) -> list:
    if name == "FeatureSim":
        return [feature_similarity_rewiring(g, k=k, train_mask=g.train_mask)
                for g in raw_graphs]
    if name == "Random":
        return [random_rewiring(g, k=k, seed=i) for i, g in enumerate(raw_graphs)]
    if name == "K-hop":
        return [khop_rewiring(g, k=k) for g in raw_graphs]
    return raw_graphs


def _apply_strategy_single(name: str, data, k: int) -> object:
    """Aplica rewiring a un único grafo (MNIST)."""
    if name == "FeatureSim":
        return feature_similarity_rewiring(data, k=k)
    if name == "Random":
        return random_rewiring(data, k=k, seed=0)
    if name == "K-hop":
        return khop_rewiring(data, k=k)
    return data


def _apply_strategy_dataset(name: str, dataset, k: int) -> list:
    """Aplica rewiring a cada grafo de un dataset PyG."""
    print(f"    Aplicando rewiring {name} (k={k}) a {len(dataset)} grafos...", flush=True)
    rewired = []
    for i, g in enumerate(dataset):
        rewired.append(_apply_strategy_single(name, g, k))
        if (i + 1) % 1000 == 0:
            print(f"    {i+1}/{len(dataset)}", flush=True)
    return rewired


# ---------------------------------------------------------------------------
# RingTransfer experiment
# ---------------------------------------------------------------------------

def run_ring(verbose: bool = True) -> tuple[dict, dict]:
    print("\n=== RingTransfer Experiment ===")
    device      = get_device()
    raw_graphs  = make_ring_transfer(**RING_CONFIG)
    baseline_ei = raw_graphs[0].edge_index.clone()

    results      = {}
    strategy_viz = {}

    for name in STRATEGIES:
        print(f"\n--- Strategy: {name} ---")
        rewired = _apply_strategy(name, raw_graphs, k=KHOP_K_RING)

        train_result = train_graph_list(
            GAT, rewired, model_kwargs=GAT_KWARGS,
            verbose=verbose, **TRAIN_RING,
        )
        test_acc = train_result['test_acc']
        model    = train_result['model']
        model.eval()
        print(f"  Test Accuracy: {test_acc:.4f}")

        test_graphs = [g for g in rewired if g.test_mask.any()]
        sample      = (test_graphs[0] if test_graphs else rewired[0]).cpu()
        model_cpu   = model.cpu()

        target_node = RING_CONFIG['num_nodes'] // 2
        source_node = 0

        print("  Computing Jacobian norms...")
        jac_by_dist = jacobian_norm_by_distance(model_cpu, sample,
                                                 num_nodes=sample.num_nodes)
        jac_mean = float(np.mean(list(jac_by_dist.values()))) if jac_by_dist else 0.0

        print("  Computing Fidelity...")
        fid_results = compute_fidelity(model_cpu, sample, [target_node])
        fid_plus    = fid_results.get(target_node, {}).get('fidelity_plus', 0.0)

        print("  Computing Sparsity...")
        sparsity = explanation_sparsity(model_cpu, sample, [target_node])

        perm = permutation_test(model_cpu, sample, [target_node], [source_node])
        print(f"  Perm test acc drop: {perm['mean_acc_drop']:.4f} ± {perm['std_acc_drop']:.4f}")

        if (name == "FeatureSim"
                and test_acc > results.get("Baseline", {}).get("test_acc", 0)
                and jac_mean < 0.01):
            print("  ⚠ ADVERTENCIA: Mejora posiblemente por sesgo, no por flujo.")

        print("  Generating attention heatmap...")
        plot_attention_heatmap(
            model=model_cpu, data=sample, strategy_name=name,
            layer_idx=-1, aggr="mean",
            save_path=f"results/ring_attention_{name.lower()}.png",
            highlight_artificial=(name != "Baseline"),
            original_edge_index=baseline_ei,
        )

        if name != "Baseline":
            summarize_attention_on_artificial(
                model_cpu, sample, baseline_ei, strategy_name=name)

        strategy_viz[name] = {
            'model': model_cpu, 'data': sample,
            'original_edge_index': baseline_ei if name != "Baseline" else None,
        }
        results[name] = {
            'test_acc': test_acc, 'jacobian_by_dist': jac_by_dist,
            'jacobian_mean': jac_mean, 'fidelity_plus': fid_plus,
            'sparsity': sparsity, 'perm_acc_drop': perm['mean_acc_drop'],
        }

    print("\n  Generating attention comparison figure...")
    plot_attention_comparison(strategy_results=strategy_viz,
                              save_path="results/ring_attention_comparison.png")
    models = {name: v['model'] for name, v in strategy_viz.items()}
    return results, models


# ---------------------------------------------------------------------------
# MNIST Superpíxeles experiment
# ---------------------------------------------------------------------------

def run_mnist(verbose: bool = True) -> tuple[dict, dict]:
    print("\n=== MNIST Superpíxeles Experiment ===")
    device = get_device()
    print(f"  Device: {device}")

    print("  Cargando dataset...")
    train_full, val_full, test_full = load_mnist_superpixels(**MNIST_CONFIG)
    baseline_ei = train_full[0].edge_index.clone()

    results      = {}
    strategy_viz = {}

    for name in STRATEGIES:
        print(f"\n--- Strategy: {name} ---")

        if name == "Baseline":
            train_ds, val_ds, test_ds = train_full, val_full, test_full
        else:
            train_ds = _apply_strategy_dataset(name, train_full, k=KHOP_K_MNIST)
            val_ds   = _apply_strategy_dataset(name, val_full,   k=KHOP_K_MNIST)
            test_ds  = _apply_strategy_dataset(name, test_full,  k=KHOP_K_MNIST)

        train_result = train_graph_level(
            GATGraphLevel,
            train_dataset=train_ds,
            val_dataset=val_ds,
            test_dataset=test_ds,
            model_kwargs=GAT_MNIST,
            verbose=verbose,
            **TRAIN_MNIST,
        )
        test_acc = train_result['test_acc']
        model    = train_result['model']
        model.eval()
        print(f"  Test Accuracy: {test_acc:.4f}")

        # Métricas de explicabilidad sobre un grafo de test representativo
        # (movemos a CPU para GNNExplainer que no soporta MPS)
        sample    = test_ds[0].cpu()
        model_cpu = model.cpu()

        print("  Computing graph-level Jacobian norms...")
        jac_by_dist = graph_level_jacobian_norm_by_distance(
            model_cpu,
            sample,
            max_reference_nodes=10,
        )

        dist_counts = distance_bucket_counts(
            sample,
            max_reference_nodes=10,
        )

        jac_mean = float(np.mean(list(jac_by_dist.values()))) if jac_by_dist else 0.0

        print(f"  Jacobian mean: {jac_mean:.6f}")
        print(f"  Distance counts: {dist_counts}")

        print("  Computing Fidelity...")
        fid_results = compute_fidelity(model_cpu, sample, [target_node],
                                       graph_level=True)
        fid_plus = fid_results.get(target_node, {}).get('fidelity_plus', 0.0)

        print("  Computing Sparsity...")
        sparsity = explanation_sparsity(model_cpu, sample, [target_node],
                                        graph_level=True)

        print("  Generating attention heatmap...")
        plot_attention_heatmap(
            model=model_cpu, data=sample,
            strategy_name=f"MNIST-{name}",
            layer_idx=-1, aggr="mean",
            save_path=f"results/mnist_attention_{name.lower()}.png",
            highlight_artificial=(name != "Baseline"),
            original_edge_index=baseline_ei,
        )

        if name != "Baseline":
            summarize_attention_on_artificial(
                model_cpu, sample, baseline_ei, strategy_name=f"MNIST-{name}")

        strategy_viz[name] = {
            'model': model_cpu, 'data': sample,
            'original_edge_index': baseline_ei if name != "Baseline" else None,
        }
        results[name] = {
            'test_acc': test_acc,
            'jacobian_by_dist': jac_by_dist,
            'jacobian_mean': jac_mean,
            'distance_counts': dist_counts,
            'fidelity_plus': fid_plus,
            'sparsity': sparsity,
            'perm_acc_drop': 0.0,
        }

    print("\n  Generating MNIST attention comparison figure...")
    plot_attention_comparison(strategy_results=strategy_viz,
                              save_path="results/mnist_attention_comparison.png")
    models = {name: v['model'] for name, v in strategy_viz.items()}
    return results, models


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LRE-GAT Over-Squashing Experiment")
    parser.add_argument('--dataset', choices=['ring', 'mnist', 'all'], default='ring')
    parser.add_argument('--quiet', action='store_true')
    args    = parser.parse_args()
    verbose = not args.quiet

    set_seed(SEED)
    print(f"[seed] Reproducibilidad fijada: {SEED}")

    device = get_device()
    print(f"[device] Usando: {device}")

    if args.dataset in ('ring', 'all'):
        ring_results, ring_models = run_ring(verbose=verbose)
        plot_report(ring_results, save_path="results/ring_report.png")
        save_results(ring_results, ring_models, prefix="ring")
        _print_summary("Ring", ring_results, include_jac=True)

    if args.dataset in ('mnist', 'all'):
        mnist_results, mnist_models = run_mnist(verbose=verbose)
        plot_report(mnist_results, save_path="results/mnist_report.png")
        save_results(mnist_results, mnist_models, prefix="mnist")
        _print_summary("MNIST", mnist_results, include_jac=False)


def _print_summary(name: str, results: dict, include_jac: bool):
    print(f"\n[{name}] Summary:")
    if include_jac:
        print(f"  {'Strategy':12s} | {'Acc':>6s} | {'JacMean':>8s} | "
              f"{'Fid+':>6s} | {'Spars':>6s} | {'PermDrop':>8s}")
        print("  " + "-" * 65)
        for s, r in results.items():
            print(f"  {s:12s} | {r['test_acc']:>6.3f} | {r['jacobian_mean']:>8.4f} | "
                  f"{r['fidelity_plus']:>6.3f} | {r['sparsity']:>6.3f} | "
                  f"{r['perm_acc_drop']:>8.4f}")
    else:
        print(f"  {'Strategy':12s} | {'Acc':>6s} | {'Fid+':>6s} | {'Spars':>6s}")
        print("  " + "-" * 42)
        for s, r in results.items():
            print(f"  {s:12s} | {r['test_acc']:>6.3f} | "
                  f"{r['fidelity_plus']:>6.3f} | {r['sparsity']:>6.3f}")


if __name__ == '__main__':
    main()
