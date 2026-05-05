"""
LRE-GAT: Squashing↔Smoothing Trade-off Experiment

Para cada (dataset, estrategia, budget):
  1. Aplica rewiring al grafo con el budget dado.
  2. Entrena un GAT desde cero.
  3. Mide test accuracy + Jacobian (squashing) + MAD/Dirichlet (smoothing).
  4. (Solo RingTransfer) calcula Fidelity+/Sparsity con GNNExplainer.

Resultados → results/trade_off.csv + plots por dataset.

Uso:
  python main.py                       # todos los datasets
  python main.py --datasets ring cora  # subset
  python main.py --quick               # 2 budgets, menos epochs (sanity check)
"""

import argparse
import os
import json
import torch
import numpy as np
import pandas as pd

from datasets import make_ring_transfer, load_mnist_superpixels, load_cora
from rewiring import STRATEGIES, apply_strategy
from models import GAT, GATGraphLevel
from train import (train_single_graph, train_graph_list, train_graph_level,
                    get_device)
from metrics import (
    jacobian_norm_by_distance,
    mean_average_distance, dirichlet_energy,
    compute_fidelity, explanation_sparsity, permutation_test,
)
from visualize import (plot_tradeoff_curves, plot_pareto_frontier,
                        plot_cross_dataset_summary, plot_attention_heatmap)


# ---------------------------------------------------------------------------
# Configuración global
# ---------------------------------------------------------------------------

BUDGETS = [0.0, 0.25, 0.75, 1.5]  # 4 budgets (0%, 25%, 75%, 150%)

RING_CONFIG = dict(num_nodes=20, num_classes=5, num_graphs=500)
GAT_RING    = dict(hidden=32, heads=4, num_layers=3, dropout=0.5)
TRAIN_RING  = dict(epochs=200, lr=0.005, weight_decay=5e-4)

GAT_CORA    = dict(hidden=16, heads=8, num_layers=2, dropout=0.6)
TRAIN_CORA  = dict(epochs=200, lr=0.005, weight_decay=5e-4)

MNIST_CONFIG = dict(subset_train=4000, subset_test=1000)  # subset reducido para tiempo
GAT_MNIST    = dict(hidden=64, heads=4, num_layers=4, dropout=0.3)
TRAIN_MNIST  = dict(epochs=25, lr=1e-3, weight_decay=5e-5, batch_size=128)


# ---------------------------------------------------------------------------
# Ejecutores por dataset
# ---------------------------------------------------------------------------

def run_ring_one(strategy: str, budget: float, verbose: bool = False) -> dict:
    """RingTransfer: lista de grafos pequeños, predicción de nodo target."""
    raw_graphs = make_ring_transfer(**RING_CONFIG)
    rewired = []
    for i, g in enumerate(raw_graphs):
        rewired.append(apply_strategy(strategy, g, budget_frac=budget,
                                       seed=i, train_mask=g.train_mask))

    res = train_graph_list(GAT, rewired, model_kwargs=GAT_RING,
                            verbose=verbose, **TRAIN_RING)
    model = res['model'].cpu().eval()

    # Métricas en un grafo de test
    test_graphs = [g for g in rewired if g.test_mask.any()]
    sample = (test_graphs[0] if test_graphs else rewired[0]).cpu()

    # Jacobian
    jbd = jacobian_norm_by_distance(model, sample)
    jac_mean = float(np.mean(list(jbd.values()))) if jbd else 0.0

    # Embeddings → MAD + Dirichlet
    emb = model.get_embeddings(sample.x, sample.edge_index)
    mad_v   = mean_average_distance(emb)
    diri_v  = dirichlet_energy(emb, sample.edge_index)

    # Explicabilidad (solo en RingTransfer)
    target = sample.num_nodes // 2
    try:
        fid = compute_fidelity(model, sample, [target])
        fid_plus = fid.get(target, {}).get('fidelity_plus', 0.0)
        spars = explanation_sparsity(model, sample, [target])
    except Exception as e:
        print(f"  [warn] fidelity/sparsity failed: {e}")
        fid_plus, spars = 0.0, 0.0

    return dict(test_acc=res['test_acc'], mad=mad_v, dirichlet=diri_v,
                jacobian_mean=jac_mean, jacobian_by_dist=jbd,
                fidelity_plus=fid_plus, sparsity=spars,
                num_nodes=sample.num_nodes,
                num_edges=int(sample.edge_index.shape[1] // 2))


def run_cora_one(strategy: str, budget: float, verbose: bool = False) -> dict:
    """Cora: un único grafo grande, clasificación de nodos."""
    data = load_cora()
    rewired = apply_strategy(strategy, data, budget_frac=budget,
                              seed=0, train_mask=data.train_mask)

    # Crea modelo
    in_ch = rewired.x.shape[1]
    out_ch = int(rewired.y.max().item()) + 1
    model_inst = GAT(in_channels=in_ch, out_channels=out_ch, **GAT_CORA)

    res = train_single_graph(model_inst, rewired, verbose=verbose, **TRAIN_CORA)
    model = res['model'].cpu().eval()
    rewired_cpu = rewired.cpu()

    # Jacobian (Cora tiene 2708 nodos → recortamos a una submuestra de 200)
    sub = _subsample_data_for_metrics(rewired_cpu, max_nodes=200, seed=0)
    jbd = jacobian_norm_by_distance(model, sub) if sub is not None else {}
    jac_mean = float(np.mean(list(jbd.values()))) if jbd else 0.0

    # MAD/Dirichlet sobre TODOS los nodos
    emb = model.get_embeddings(rewired_cpu.x, rewired_cpu.edge_index)
    mad_v  = mean_average_distance(emb)
    diri_v = dirichlet_energy(emb, rewired_cpu.edge_index)

    # Fidelity en algunos nodos de test (caro → solo 5)
    test_idx = rewired_cpu.test_mask.nonzero(as_tuple=True)[0]
    samples = test_idx[torch.randperm(len(test_idx))[:5]].tolist()
    try:
        fid = compute_fidelity(model, rewired_cpu, samples)
        fid_plus = float(np.mean([v.get('fidelity_plus', 0.0)
                                  for v in fid.values()]))
        spars = explanation_sparsity(model, rewired_cpu, samples)
    except Exception as e:
        print(f"  [warn] fidelity/sparsity failed: {e}")
        fid_plus, spars = 0.0, 0.0

    return dict(test_acc=res['test_acc'], mad=mad_v, dirichlet=diri_v,
                jacobian_mean=jac_mean, jacobian_by_dist=jbd,
                fidelity_plus=fid_plus, sparsity=spars,
                num_nodes=rewired_cpu.num_nodes,
                num_edges=int(rewired_cpu.edge_index.shape[1] // 2))


def run_mnist_one(strategy: str, budget: float, verbose: bool = False) -> dict:
    """MNIST Superpíxeles: dataset de grafos, clasificación a nivel de grafo."""
    train_full, val_full, test_full = load_mnist_superpixels(**MNIST_CONFIG)

    if strategy == "Baseline" or budget == 0.0:
        train_ds, val_ds, test_ds = train_full, val_full, test_full
    else:
        print(f"    Aplicando rewiring {strategy} budget={budget}...", flush=True)
        train_ds = [apply_strategy(strategy, g, budget_frac=budget, seed=i)
                    for i, g in enumerate(train_full)]
        val_ds   = [apply_strategy(strategy, g, budget_frac=budget, seed=i)
                    for i, g in enumerate(val_full)]
        test_ds  = [apply_strategy(strategy, g, budget_frac=budget, seed=i)
                    for i, g in enumerate(test_full)]

    res = train_graph_level(GATGraphLevel,
                              train_dataset=train_ds, val_dataset=val_ds,
                              test_dataset=test_ds, model_kwargs=GAT_MNIST,
                              verbose=verbose, **TRAIN_MNIST)
    model = res['model'].cpu().eval()

    sample = test_ds[0].cpu() if not isinstance(test_ds, list) else test_ds[0].cpu()
    if hasattr(sample, 'cpu'):
        sample = sample.cpu()

    # MAD/Dirichlet sobre embeddings de un grafo
    emb = model.get_embeddings(sample.x, sample.edge_index)
    mad_v  = mean_average_distance(emb)
    diri_v = dirichlet_energy(emb, sample.edge_index)

    # Jacobian a nivel de grafo: una pasada
    try:
        jbd = jacobian_norm_by_distance(model, sample, graph_level=True)
        jac_mean = float(np.mean(list(jbd.values()))) if jbd else 0.0
    except Exception as e:
        print(f"  [warn] jacobian failed: {e}")
        jbd, jac_mean = {}, 0.0

    # En MNIST no calculamos Fidelity (multi-grafo, demasiado caro)
    return dict(test_acc=res['test_acc'], mad=mad_v, dirichlet=diri_v,
                jacobian_mean=jac_mean, jacobian_by_dist=jbd,
                fidelity_plus=0.0, sparsity=0.0,
                num_nodes=sample.num_nodes,
                num_edges=int(sample.edge_index.shape[1] // 2))


# ---------------------------------------------------------------------------
# Helper: submuestreo de Cora para Jacobian
# ---------------------------------------------------------------------------

def _subsample_data_for_metrics(data, max_nodes: int = 200, seed: int = 0):
    """
    Para grafos grandes (Cora): toma una submuestra inducida de ~max_nodes nodos.
    Devuelve un Data más pequeño donde podemos calcular distancias y Jacobian.
    """
    if data.num_nodes <= max_nodes:
        return data
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(data.num_nodes, generator=g)[:max_nodes]
    idx_set = set(idx.tolist())
    # Reindex
    remap = {old: new for new, old in enumerate(idx.tolist())}
    src = data.edge_index[0].tolist()
    dst = data.edge_index[1].tolist()
    new_src, new_dst = [], []
    for s, d in zip(src, dst):
        if s in idx_set and d in idx_set:
            new_src.append(remap[s])
            new_dst.append(remap[d])
    if len(new_src) == 0:
        return None
    from torch_geometric.data import Data as PyGData
    return PyGData(
        x=data.x[idx],
        edge_index=torch.tensor([new_src, new_dst], dtype=torch.long),
        y=data.y[idx],
    )


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------

DATASET_RUNNERS = {
    "ring":  run_ring_one,
    "cora":  run_cora_one,
    "mnist": run_mnist_one,
}


def run_sweep(datasets: list, budgets: list, verbose: bool) -> pd.DataFrame:
    rows = []
    total = sum(len(STRATEGIES) * len(budgets) for _ in datasets)
    print(f"\n=== Sweep: {len(datasets)} datasets × {len(STRATEGIES)} strategies × "
          f"{len(budgets)} budgets = {total} runs ===\n")

    run_idx = 0
    for ds in datasets:
        if ds not in DATASET_RUNNERS:
            print(f"[skip] unknown dataset: {ds}")
            continue
        runner = DATASET_RUNNERS[ds]

        for strategy in STRATEGIES:
            for budget in budgets:
                # Baseline solo se evalúa en budget=0 (idéntico a otros budgets para Baseline)
                if strategy == "Baseline" and budget != 0.0:
                    continue

                run_idx += 1
                tag = f"[{run_idx}] {ds} | {strategy} | b={budget}"
                print(f"\n{tag}", flush=True)
                try:
                    out = runner(strategy=strategy, budget=budget, verbose=verbose)
                    out.update(dict(dataset=ds, strategy=strategy, budget=budget))
                    rows.append(out)
                    print(f"  → acc={out['test_acc']:.3f} | "
                          f"MAD={out['mad']:.3f} | jac={out['jacobian_mean']:.4f} | "
                          f"E={out['num_edges']}", flush=True)
                except Exception as e:
                    print(f"  ✗ FAILED: {e}", flush=True)
                    import traceback; traceback.print_exc()
                    rows.append(dict(dataset=ds, strategy=strategy, budget=budget,
                                      test_acc=0.0, mad=0.0, dirichlet=0.0,
                                      jacobian_mean=0.0, jacobian_by_dist={},
                                      fidelity_plus=0.0, sparsity=0.0,
                                      num_nodes=0, num_edges=0,
                                      error=str(e)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Replicar Baseline para todos los budgets (para que las plots tengan línea recta)
# ---------------------------------------------------------------------------

def expand_baseline(df: pd.DataFrame, budgets: list) -> pd.DataFrame:
    extra = []
    for ds in df['dataset'].unique():
        base_row = df[(df['dataset'] == ds) & (df['strategy'] == "Baseline")]
        if base_row.empty:
            continue
        base = base_row.iloc[0].to_dict()
        for b in budgets:
            if b == 0.0:
                continue
            new = dict(base); new['budget'] = b
            extra.append(new)
    return pd.concat([df, pd.DataFrame(extra)], ignore_index=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LRE-GAT trade-off study")
    parser.add_argument('--datasets', nargs='+',
                         default=['ring', 'cora', 'mnist'],
                         choices=['ring', 'cora', 'mnist'])
    parser.add_argument('--quick', action='store_true',
                         help="Sanity check: 2 budgets, less epochs")
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    print(f"[device] {get_device()}")

    budgets = [0.0, 0.5] if args.quick else BUDGETS
    if args.quick:
        TRAIN_RING['epochs']  = 60
        TRAIN_CORA['epochs']  = 80
        TRAIN_MNIST['epochs'] = 8

    # ---- Sweep ---- #
    df = run_sweep(args.datasets, budgets, verbose=not args.quiet)
    df = expand_baseline(df, budgets)

    # ---- Persistencia ---- #
    csv_path = "results/trade_off.csv"
    df_to_save = df.drop(columns=['jacobian_by_dist'], errors='ignore')
    df_to_save.to_csv(csv_path, index=False)
    print(f"\n[save] CSV → {csv_path}")

    json_path = "results/trade_off_full.json"
    df_records = df.copy()
    df_records['jacobian_by_dist'] = df_records['jacobian_by_dist'].apply(
        lambda d: {int(k): float(v) for k, v in d.items()} if isinstance(d, dict) else {})
    with open(json_path, 'w') as f:
        json.dump(df_records.to_dict(orient='records'), f, indent=2, default=str)
    print(f"[save] JSON → {json_path}")

    # ---- Plots ---- #
    for ds in args.datasets:
        plot_tradeoff_curves(df, dataset_name=ds,
                              save_path=f"results/tradeoff_{ds}.png")
        plot_pareto_frontier(df, dataset_name=ds,
                              save_path=f"results/pareto_{ds}.png")
    plot_cross_dataset_summary(df, save_path="results/summary.png")

    # ---- Resumen consola ---- #
    print("\n" + "=" * 70)
    print("FINAL SUMMARY  (best accuracy per dataset × strategy)")
    print("=" * 70)
    best = (df.groupby(['dataset', 'strategy'])
              .agg({'test_acc': 'max'}).reset_index())
    pivot = best.pivot(index='dataset', columns='strategy', values='test_acc')
    print(pivot.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == '__main__':
    main()
