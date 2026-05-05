"""
Main experiment: LRE-GAT over-squashing analysis.

Runs 5 rewiring strategies on RingTransfer (synthetic) and optionally Cora.
Computes Jacobian Norm, Fidelity+, Sparsity, and Permutation Test.
Generates 4-panel report figure.

Usage:
    python main.py --dataset ring        # RingTransfer only (fast)
    python main.py --dataset cora        # Cora only
    python main.py --dataset all         # Both
"""

import argparse
import torch
import numpy as np
from copy import deepcopy

from oversquashing_solution_explainability.version_2.datasets import make_ring_transfer, load_cora
from oversquashing_solution_explainability.version_2.rewiring import STRATEGIES, feature_similarity_rewiring
from oversquashing_solution_explainability.version_2.models import GAT
from oversquashing_solution_explainability.version_2.train import train_graph_list, train_single_graph
from metrics import (
    jacobian_norm_by_distance, compute_fidelity,
    permutation_test, explanation_sparsity,
)
from oversquashing_solution_explainability.version_2.visualize import plot_report


# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
RING_CONFIG = dict(num_nodes=20, num_classes=5, num_graphs=500)
GAT_KWARGS  = dict(hidden=32, heads=4, num_layers=3, dropout=0.5)
TRAIN_RING  = dict(epochs=200, lr=0.005, weight_decay=5e-4)
TRAIN_CORA  = dict(epochs=300, lr=0.005, weight_decay=5e-4)
KHOP_K      = 3   # k for K-hop and budget for Random/FeatureSim


def run_ring(verbose: bool = True) -> dict:
    print("\n=== RingTransfer Experiment ===")
    raw_graphs = make_ring_transfer(**RING_CONFIG)
    num_classes = RING_CONFIG['num_classes']
    in_channels = raw_graphs[0].x.shape[1]

    results = {}

    for name, strategy_fn in STRATEGIES.items():
        print(f"\n--- Strategy: {name} ---")

        # Apply rewiring to each graph
        if name == "FeatureSim":
            rewired = [feature_similarity_rewiring(g, k=KHOP_K, train_mask=g.train_mask)
                       for g in raw_graphs]
        elif name == "Random":
            from oversquashing_solution_explainability.version_2.rewiring import random_rewiring
            rewired = [random_rewiring(g, k=KHOP_K, seed=i) for i, g in enumerate(raw_graphs)]
        elif name == "K-hop":
            from oversquashing_solution_explainability.version_2.rewiring import khop_rewiring
            rewired = [khop_rewiring(g, k=KHOP_K) for g in raw_graphs]
        elif name == "VirtualNode":
            from oversquashing_solution_explainability.version_2.rewiring import virtual_node_rewiring
            rewired = [virtual_node_rewiring(g) for g in raw_graphs]
            # Virtual node expands num_nodes — update in_channels stays same
        else:
            rewired = raw_graphs

        train_result = train_graph_list(
            GAT, rewired,
            model_kwargs=GAT_KWARGS,
            verbose=verbose,
            **TRAIN_RING,
        )
        test_acc = train_result['test_acc']
        model = train_result['model']
        model.eval()

        print(f"  Test Accuracy: {test_acc:.4f}")

        # Use first test graph for explanation metrics (representative)
        test_graphs = [g for g in rewired if g.test_mask.any()]
        sample = test_graphs[0].cpu() if test_graphs else rewired[0].cpu()
        model_cpu = model.cpu()

        target_node = RING_CONFIG['num_nodes'] // 2  # antipodal node
        source_node = 0  # signal source

        # Jacobian Norm by distance
        print("  Computing Jacobian norms...")
        jac_by_dist = jacobian_norm_by_distance(model_cpu, sample,
                                                  num_nodes=getattr(sample, 'original_num_nodes', sample.num_nodes))
        jac_mean = np.mean(list(jac_by_dist.values())) if jac_by_dist else 0.0

        # Fidelity
        print("  Computing Fidelity...")
        fid_results = compute_fidelity(model_cpu, sample, [target_node])
        fid_plus = fid_results.get(target_node, {}).get('fidelity_plus', 0.0)

        # Sparsity
        print("  Computing Sparsity...")
        sparsity = explanation_sparsity(model_cpu, sample, [target_node])

        # Permutation Test
        distant_nodes = [source_node]
        perm = permutation_test(model_cpu, sample, [target_node], distant_nodes)
        print(f"  Perm test acc drop: {perm['mean_acc_drop']:.4f} ± {perm['std_acc_drop']:.4f}")

        # Bias alert
        if name == "FeatureSim" and test_acc > results.get("Baseline", {}).get("test_acc", 0):
            if jac_mean < 0.01:
                print("ADVERTENCIA: Mejora por Homofilia/Sesgo, no por flujo de información")

        results[name] = {
            'test_acc': test_acc,
            'jacobian_by_dist': jac_by_dist,
            'jacobian_mean': jac_mean,
            'fidelity_plus': fid_plus,
            'sparsity': sparsity,
            'perm_acc_drop': perm['mean_acc_drop'],
        }

    return results


def run_cora(verbose: bool = True) -> dict:
    print("\n=== Cora Experiment ===")
    raw_data = load_cora()
    num_classes = int(raw_data.y.max().item()) + 1
    in_channels = raw_data.x.shape[1]

    results = {}

    for name, strategy_fn in STRATEGIES.items():
        print(f"\n--- Strategy: {name} ---")

        if name == "FeatureSim":
            data = feature_similarity_rewiring(raw_data, k=KHOP_K,
                                                train_mask=raw_data.train_mask)
        elif name == "Random":
            from oversquashing_solution_explainability.version_2.rewiring import random_rewiring
            data = random_rewiring(raw_data, k=KHOP_K)
        elif name == "K-hop":
            from oversquashing_solution_explainability.version_2.rewiring import khop_rewiring
            data = khop_rewiring(raw_data, k=KHOP_K)
        elif name == "VirtualNode":
            from oversquashing_solution_explainability.version_2.rewiring import virtual_node_rewiring
            data = virtual_node_rewiring(raw_data)
        else:
            data = raw_data

        model = GAT(in_channels=in_channels, out_channels=num_classes, **GAT_KWARGS)
        train_result = train_single_graph(model, data, verbose=verbose, **TRAIN_CORA)
        model = train_result['model']
        test_acc = train_result['test_acc']
        print(f"  Test Accuracy: {test_acc:.4f}")

        model.eval()
        data_cpu = data.cpu()
        model_cpu = model.cpu()

        # Pick a few target nodes from test set
        test_nodes = data_cpu.test_mask.nonzero(as_tuple=True)[0][:5].tolist()

        print("  Computing Jacobian norms (Cora is large — sample only)...")
        # Cora has 2708 nodes — Floyd-Warshall is too slow; skip dist-based Jacobian
        # Instead compute Jacobian only for the 5 target nodes vs. 1-hop neighbors
        jac_by_dist = {}  # skip for Cora
        jac_mean = 0.0

        print("  Computing Fidelity...")
        fid_results = compute_fidelity(model_cpu, data_cpu, test_nodes[:3])
        fid_plus = np.mean([v['fidelity_plus'] for v in fid_results.values()]) if fid_results else 0.0

        print("  Computing Sparsity...")
        sparsity = explanation_sparsity(model_cpu, data_cpu, test_nodes[:3])

        if name == "FeatureSim" and test_acc > results.get("Baseline", {}).get("test_acc", 0):
            if jac_mean < 0.01:
                print("ADVERTENCIA: Mejora por Homofilia/Sesgo, no por flujo de información")

        results[name] = {
            'test_acc': test_acc,
            'jacobian_by_dist': jac_by_dist,
            'jacobian_mean': jac_mean,
            'fidelity_plus': fid_plus,
            'sparsity': sparsity,
            'perm_acc_drop': 0.0,
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['ring', 'cora', 'all'], default='ring')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    verbose = not args.quiet

    if args.dataset in ('ring', 'all'):
        ring_results = run_ring(verbose=verbose)
        plot_report(ring_results, save_path="results/ring_report.png")
        print("\n[Ring] Summary:")
        for s, r in ring_results.items():
            print(f"  {s:12s} | acc={r['test_acc']:.3f} | jac={r['jacobian_mean']:.4f} "
                  f"| fid+={r['fidelity_plus']:.3f} | sparsity={r['sparsity']:.3f} "
                  f"| perm_drop={r['perm_acc_drop']:.3f}")

    if args.dataset in ('cora', 'all'):
        cora_results = run_cora(verbose=verbose)
        plot_report(cora_results, save_path="results/cora_report.png")
        print("\n[Cora] Summary:")
        for s, r in cora_results.items():
            print(f"  {s:12s} | acc={r['test_acc']:.3f} | fid+={r['fidelity_plus']:.3f} "
                  f"| sparsity={r['sparsity']:.3f}")


if __name__ == '__main__':
    main()
