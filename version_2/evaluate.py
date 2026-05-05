"""
evaluate.py — Load saved models and compute all metrics without retraining.

Usage:
    python evaluate.py --dataset ring       # RingTransfer only
    python evaluate.py --dataset mnist      # MNIST only
    python evaluate.py --dataset all        # Both

Expects results/ folder with:
    ring_model_baseline.pt, ring_model_k-hop.pt, ring_model_random.pt,
    ring_model_featuresim.pt
    mnist_model_baseline.pt, mnist_model_k-hop.pt, etc.
"""

import argparse
import json
import os
import torch
import numpy as np

from datasets import make_ring_transfer, load_mnist_superpixels #type: ignore
from rewiring import random_rewiring, khop_rewiring, feature_similarity_rewiring
from models import GAT, GATGraphLevel #type: ignore
from metrics import (
    jacobian_norm_by_distance,
    graph_level_jacobian_norm_by_distance,
    distance_bucket_counts,
    compute_jacobian_norm,
    compute_fidelity,
    permutation_test,
    explanation_sparsity,
)
from visualize import plot_report


# ---------------------------------------------------------------------------
# Must match the config used during training
# ---------------------------------------------------------------------------
RING_CONFIG = dict(num_nodes=20, num_classes=5, num_graphs=500)
GAT_KWARGS  = dict(hidden=32, heads=4, num_layers=3, dropout=0.5)

MNIST_CONFIG = dict(subset_train=10000, subset_test=2000)
GAT_MNIST    = dict(hidden=64, heads=4, num_layers=4, dropout=0.3)
KHOP_K_RING  = 3
KHOP_K_MNIST = 3

STRATEGIES_RING  = ["Baseline", "Random", "K-hop", "FeatureSim"]
STRATEGIES_MNIST = ["Baseline", "Random", "K-hop", "FeatureSim"]

MODEL_NAME_MAP = {
    "Baseline":   "baseline",
    "Random":     "random",
    "K-hop":      "k-hop",
    "FeatureSim": "featuresim",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(model_cls, model_kwargs, path: str, in_channels: int,
               out_channels: int) -> torch.nn.Module:
    model = model_cls(in_channels=in_channels, out_channels=out_channels,
                      **model_kwargs)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def apply_rewiring(name: str, graphs: list, k: int) -> list:
    if name == "Random":
        return [random_rewiring(g, k=k, seed=i) for i, g in enumerate(graphs)]
    if name == "K-hop":
        return [khop_rewiring(g, k=k) for g in graphs]
    if name == "FeatureSim":
        return [feature_similarity_rewiring(g, k=k, train_mask=None) #type: ignore
                for g in graphs]
    return graphs  # Baseline


def apply_rewiring_single(name: str, data, k: int):
    if name == "Random":
        return random_rewiring(data, k=k, seed=0)
    if name == "K-hop":
        return khop_rewiring(data, k=k)
    if name == "FeatureSim":
        return feature_similarity_rewiring(data, k=k)
    return data


def _print_table(results: dict, include_jac: bool = True):
    if include_jac:
        print(f"\n  {'Strategy':12s} | {'Acc':>6s} | {'JacMean':>8s} | "
              f"{'Fid+':>6s} | {'Spars':>6s} | {'PermDrop':>8s}")
        print("  " + "-" * 65)
        for s, r in results.items():
            print(f"  {s:12s} | {r['test_acc']:>6.3f} | {r['jacobian_mean']:>8.4f} | "
                  f"{r['fidelity_plus']:>6.3f} | {r['sparsity']:>6.3f} | "
                  f"{r['perm_acc_drop']:>8.4f}")
    else:
        print(f"\n  {'Strategy':12s} | {'Acc':>6s} | {'Fid+':>6s} | {'Spars':>6s}")
        print("  " + "-" * 42)
        for s, r in results.items():
            print(f"  {s:12s} | {r['test_acc']:>6.3f} | "
                  f"{r['fidelity_plus']:>6.3f} | {r['sparsity']:>6.3f}")


# ---------------------------------------------------------------------------
# RingTransfer evaluation
# ---------------------------------------------------------------------------

def evaluate_ring():
    print("\n=== Evaluating RingTransfer ===")
    raw_graphs  = make_ring_transfer(**RING_CONFIG)
    num_classes = RING_CONFIG['num_classes']
    in_channels = raw_graphs[0].x.shape[1] #type: ignore

    # Load previously saved test_acc from JSON (avoids re-running eval loop)
    saved_metrics = {}
    json_path = "results/ring_metrics.json"
    if os.path.exists(json_path):
        with open(json_path) as f:
            saved_metrics = json.load(f)
        print(f"  Loaded saved metrics from {json_path}")

    results = {}

    for name in STRATEGIES_RING:
        safe_name = MODEL_NAME_MAP[name]
        model_path = f"results/ring_model_{safe_name}.pt"

        if not os.path.exists(model_path):
            print(f"  [skip] {model_path} not found — skipping {name}")
            continue

        print(f"\n--- {name} ---")
        model = load_model(GAT, GAT_KWARGS, model_path, in_channels, num_classes)

        # Reconstruct rewired graphs (RingTransfer is deterministic)
        rewired = apply_rewiring(name, raw_graphs, k=KHOP_K_RING)

        test_graphs = [g for g in rewired if g.test_mask.any()]
        sample = (test_graphs[0] if test_graphs else rewired[0]).cpu()

        target_node = RING_CONFIG['num_nodes'] // 2
        source_node = 0

        # Recover test_acc from saved JSON if available
        test_acc = saved_metrics.get(name, {}).get('test_acc', float('nan'))
        if not np.isnan(test_acc):
            print(f"  test_acc (from saved): {test_acc:.4f}")
        else:
            print("  test_acc not found in saved metrics — run main.py first")

        print("  Computing Jacobian norms...")
        jac_by_dist = jacobian_norm_by_distance(model, sample,
                                                 num_nodes=sample.num_nodes)
        jac_mean = float(np.mean(list(jac_by_dist.values()))) if jac_by_dist else 0.0
        print(f"  Jacobian mean: {jac_mean:.6f}")

        print("  Computing Fidelity...")
        fid_results = compute_fidelity(model, sample, [target_node])
        fid_plus = fid_results.get(target_node, {}).get('fidelity_plus', 0.0)
        print(f"  Fidelity+: {fid_plus:.4f}")

        print("  Computing Sparsity...")
        sparsity = explanation_sparsity(model, sample, [target_node])
        print(f"  Sparsity: {sparsity:.4f}")

        print("  Computing Permutation test...")
        perm = permutation_test(model, sample, [target_node], [source_node])
        print(f"  Perm acc drop: {perm['mean_acc_drop']:.4f} ± {perm['std_acc_drop']:.4f}")

        results[name] = {
            'test_acc':        test_acc,
            'jacobian_by_dist': jac_by_dist,
            'jacobian_mean':   jac_mean,
            'fidelity_plus':   fid_plus,
            'sparsity':        sparsity,
            'perm_acc_drop':   perm['mean_acc_drop'],
        }

    if results:
        print("\n[Ring] Summary:")
        _print_table(results, include_jac=True)
        plot_report(results, save_path="results/ring_eval_report.png")
        print("\n  Report saved → results/ring_eval_report.png")

    return results


# ---------------------------------------------------------------------------
# MNIST evaluation
# ---------------------------------------------------------------------------

def evaluate_mnist():
    print("\n=== Evaluating MNIST Superpixels ===")
    print("  Loading dataset (this may take a moment)...")
    train_ds, val_ds, test_ds = load_mnist_superpixels(**MNIST_CONFIG)

    in_channels = train_ds[0].x.shape[1]
    all_labels  = [int(d.y.item()) for d in train_ds] + \
                  [int(d.y.item()) for d in val_ds] + \
                  [int(d.y.item()) for d in test_ds]
    num_classes = max(all_labels) + 1

    saved_metrics = {}
    json_path = "results/mnist_metrics.json"
    if os.path.exists(json_path):
        with open(json_path) as f:
            saved_metrics = json.load(f)
        print(f"  Loaded saved metrics from {json_path}")

    results = {}

    for name in STRATEGIES_MNIST:
        safe_name  = MODEL_NAME_MAP[name]
        model_path = f"results/mnist_model_{safe_name}.pt"

        if not os.path.exists(model_path):
            print(f"  [skip] {model_path} not found — skipping {name}")
            continue

        print(f"\n--- {name} ---")
        model = load_model(GATGraphLevel, GAT_MNIST, model_path,
                           in_channels, num_classes)

        # Use one representative test graph (rewired the same way)
        sample_raw  = test_ds[0]
        sample      = apply_rewiring_single(name, sample_raw, k=KHOP_K_MNIST).cpu()
        target_node = sample.num_nodes // 2  # used by fidelity & sparsity

        print("  Computing graph-level Jacobian norms...")
        jac_by_dist = graph_level_jacobian_norm_by_distance(
            model,
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

        test_acc = saved_metrics.get(name, {}).get('test_acc', float('nan'))
        if not np.isnan(test_acc):
            print(f"  test_acc (from saved): {test_acc:.4f}")

        print("  Computing Fidelity...")
        fid_results = compute_fidelity(model, sample, [target_node],
                                       graph_level=True) #type: ignore
        fid_plus = fid_results.get(target_node, {}).get('fidelity_plus', 0.0)
        print(f"  Fidelity+: {fid_plus:.4f}")

        print("  Computing Sparsity...")
        sparsity = explanation_sparsity(model, sample, [target_node],
                                        graph_level=True) #type: ignore
        print(f"  Sparsity: {sparsity:.4f}")

        results[name] = {
            'test_acc':         test_acc,
            'jacobian_by_dist': jac_by_dist,
            'jacobian_mean':    jac_mean,
            'distance_counts':  dist_counts,
            'fidelity_plus':    fid_plus,
            'sparsity':         sparsity,
            'perm_acc_drop':    0.0,
        }

    if results:
        print("\n[MNIST] Summary:")
        _print_table(results, include_jac=False)
        plot_report(results, save_path="results/mnist_eval_report.png")
        print("\n  Report saved → results/mnist_eval_report.png")

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate saved LRE-GAT models")
    parser.add_argument('--dataset', choices=['ring', 'mnist', 'all'], default='ring')
    args = parser.parse_args()

    if args.dataset in ('ring', 'all'):
        evaluate_ring()

    if args.dataset in ('mnist', 'all'):
        evaluate_mnist()


if __name__ == '__main__':
    main()