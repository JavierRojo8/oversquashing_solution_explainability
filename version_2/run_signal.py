"""
run_signal.py — Signal-Guided Rewiring experiment runner.

Two-phase protocol per (dataset, strategy, budget):
  Phase A: train a GAT baseline on the ORIGINAL graph.
  Phase B: extract Jacobian/attention signal from that baseline → apply
           rewiring → retrain from scratch → measure the same metrics
           as main.py.

Supported datasets: ring, cora  (MNIST excluded in v1 — too expensive per-graph).
Results are appended to results/trade_off.csv and results/trade_off_full.json
so they can be compared directly with existing strategy results.

Usage:
  python run_signal.py                          # ring + cora, all budgets
  python run_signal.py --datasets ring          # ring only
  python run_signal.py --budgets 0.25 --quiet   # quick sanity check
  python run_signal.py --strategies JacobianGuided
"""

import argparse
import json
import os
import torch
import numpy as np
import pandas as pd

from datasets import make_ring_transfer, load_cora
from models import GAT
from train import train_graph_list, train_single_graph, get_device
from metrics import (
    jacobian_norm_by_distance,
    mean_average_distance,
    dirichlet_energy,
    compute_fidelity,
    explanation_sparsity,
)
from main import (
    RING_CONFIG, GAT_RING, TRAIN_RING,
    GAT_CORA, TRAIN_CORA,
    _subsample_data_for_metrics,
)
from signal_rewiring import jacobian_guided_rewiring, attention_guided_rewiring
from visualize import plot_tradeoff_curves, plot_pareto_frontier


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BUDGETS           = [0.25, 0.75, 1.5]
SIGNAL_STRATEGIES = ["JacobianGuided", "AttentionGuided"]
CSV_PATH          = "results/trade_off.csv"
JSON_PATH         = "results/trade_off_full.json"


# ---------------------------------------------------------------------------
# Strategy dispatch
# ---------------------------------------------------------------------------

def _apply_signal(strategy: str, model, data, budget: float,
                  graph_level: bool, seed: int):
    if strategy == "JacobianGuided":
        return jacobian_guided_rewiring(
            model, data, budget_frac=budget,
            graph_level=graph_level, seed=seed,
        )
    if strategy == "AttentionGuided":
        return attention_guided_rewiring(
            model, data, budget_frac=budget,
            graph_level=graph_level, seed=seed,
        )
    raise ValueError(f"Unknown signal strategy: {strategy}")


# ---------------------------------------------------------------------------
# Dataset runners
# ---------------------------------------------------------------------------

def run_signal_ring_one(strategy: str, budget: float, verbose: bool = False) -> dict:
    """RingTransfer two-phase run: baseline → signal rewiring → retrain."""
    raw_graphs = make_ring_transfer(**RING_CONFIG)

    # Phase A: train baseline on original graphs
    print("  [Phase A] Training baseline on original ring graphs...")
    res_base = train_graph_list(
        GAT, raw_graphs, model_kwargs=GAT_RING,
        verbose=verbose, **TRAIN_RING,
    )
    base_model = res_base['model'].cpu().eval()
    print(f"  [Phase A] Baseline test_acc={res_base['test_acc']:.4f}")

    # Apply signal rewiring to each graph individually
    print(f"  Applying {strategy} rewiring to {len(raw_graphs)} graphs...")
    rewired = []
    for i, g in enumerate(raw_graphs):
        rewired.append(_apply_signal(
            strategy, base_model, g.cpu(), budget,
            graph_level=False, seed=i,
        ))

    # Phase B: retrain from scratch on rewired graphs
    print("  [Phase B] Retraining from scratch on rewired graphs...")
    res = train_graph_list(
        GAT, rewired, model_kwargs=GAT_RING,
        verbose=verbose, **TRAIN_RING,
    )
    model = res['model'].cpu().eval()

    # Metrics (same as run_ring_one in main.py)
    test_graphs = [g for g in rewired if g.test_mask.any()]
    sample      = (test_graphs[0] if test_graphs else rewired[0]).cpu()

    jbd      = jacobian_norm_by_distance(model, sample)
    jac_mean = float(np.mean(list(jbd.values()))) if jbd else 0.0

    emb    = model.get_embeddings(sample.x, sample.edge_index)
    mad_v  = mean_average_distance(emb)
    diri_v = dirichlet_energy(emb, sample.edge_index)

    target = sample.num_nodes // 2
    try:
        fid      = compute_fidelity(model, sample, [target])
        fid_plus = fid.get(target, {}).get('fidelity_plus', 0.0)
        spars    = explanation_sparsity(model, sample, [target])
    except Exception as e:
        print(f"  [warn] fidelity/sparsity failed: {e}")
        fid_plus, spars = 0.0, 0.0

    return dict(
        test_acc=res['test_acc'], mad=mad_v, dirichlet=diri_v,
        jacobian_mean=jac_mean, jacobian_by_dist=jbd,
        fidelity_plus=fid_plus, sparsity=spars,
        num_nodes=sample.num_nodes,
        num_edges=int(sample.edge_index.shape[1] // 2),
    )


def run_signal_cora_one(strategy: str, budget: float, verbose: bool = False) -> dict:
    """Cora two-phase run: baseline → signal rewiring → retrain."""
    data  = load_cora()
    in_ch  = data.x.shape[1]
    out_ch = int(data.y.max().item()) + 1

    # Phase A: train baseline on original Cora
    print("  [Phase A] Training baseline on original Cora...")
    base_model_inst = GAT(in_channels=in_ch, out_channels=out_ch, **GAT_CORA)
    res_base = train_single_graph(
        base_model_inst, data, verbose=verbose, **TRAIN_CORA,
    )
    base_model = res_base['model'].cpu().eval()
    print(f"  [Phase A] Baseline test_acc={res_base['test_acc']:.4f}")

    # Apply signal rewiring
    print(f"  Applying {strategy} rewiring to Cora...")
    rewired = _apply_signal(
        strategy, base_model, data.cpu(), budget,
        graph_level=False, seed=0,
    )

    # Phase B: retrain from scratch
    print("  [Phase B] Retraining from scratch on rewired Cora...")
    fresh_model = GAT(in_channels=in_ch, out_channels=out_ch, **GAT_CORA)
    res = train_single_graph(fresh_model, rewired, verbose=verbose, **TRAIN_CORA)
    model       = res['model'].cpu().eval()
    rewired_cpu = rewired.cpu()

    # Metrics (same as run_cora_one in main.py)
    sub      = _subsample_data_for_metrics(rewired_cpu, max_nodes=200, seed=0)
    jbd      = jacobian_norm_by_distance(model, sub) if sub is not None else {}
    jac_mean = float(np.mean(list(jbd.values()))) if jbd else 0.0

    emb    = model.get_embeddings(rewired_cpu.x, rewired_cpu.edge_index)
    mad_v  = mean_average_distance(emb)
    diri_v = dirichlet_energy(emb, rewired_cpu.edge_index)

    test_idx     = rewired_cpu.test_mask.nonzero(as_tuple=True)[0]
    sample_nodes = test_idx[torch.randperm(len(test_idx))[:5]].tolist()
    try:
        fid      = compute_fidelity(model, rewired_cpu, sample_nodes)
        fid_plus = float(np.mean([v.get('fidelity_plus', 0.0)
                                   for v in fid.values()]))
        spars    = explanation_sparsity(model, rewired_cpu, sample_nodes)
    except Exception as e:
        print(f"  [warn] fidelity/sparsity failed: {e}")
        fid_plus, spars = 0.0, 0.0

    return dict(
        test_acc=res['test_acc'], mad=mad_v, dirichlet=diri_v,
        jacobian_mean=jac_mean, jacobian_by_dist=jbd,
        fidelity_plus=fid_plus, sparsity=spars,
        num_nodes=rewired_cpu.num_nodes,
        num_edges=int(rewired_cpu.edge_index.shape[1] // 2),
    )


# ---------------------------------------------------------------------------
# Sweep orchestration
# ---------------------------------------------------------------------------

SIGNAL_RUNNERS = {
    "ring": run_signal_ring_one,
    "cora": run_signal_cora_one,
}


def run_signal_sweep(datasets: list, strategies: list, budgets: list,
                     verbose: bool) -> pd.DataFrame:
    total = len(datasets) * len(strategies) * len(budgets)
    print(f"\n=== Signal sweep: {len(datasets)} datasets × "
          f"{len(strategies)} strategies × {len(budgets)} budgets "
          f"= {total} runs ===\n")

    rows    = []
    run_idx = 0
    for ds in datasets:
        if ds not in SIGNAL_RUNNERS:
            print(f"[skip] unsupported dataset for signal rewiring: {ds}")
            continue
        runner = SIGNAL_RUNNERS[ds]
        for strategy in strategies:
            for budget in budgets:
                run_idx += 1
                tag = f"[{run_idx}] {ds} | {strategy} | b={budget}"
                print(f"\n{tag}", flush=True)
                try:
                    out = runner(strategy=strategy, budget=budget, verbose=verbose)
                    out.update(dict(dataset=ds, strategy=strategy, budget=budget))
                    rows.append(out)
                    print(f"  → acc={out['test_acc']:.3f} | "
                          f"MAD={out['mad']:.3f} | "
                          f"jac={out['jacobian_mean']:.4f} | "
                          f"E={out['num_edges']}", flush=True)
                except Exception as e:
                    print(f"  FAILED: {e}", flush=True)
                    import traceback; traceback.print_exc()
                    rows.append(dict(
                        dataset=ds, strategy=strategy, budget=budget,
                        test_acc=0.0, mad=0.0, dirichlet=0.0,
                        jacobian_mean=0.0, jacobian_by_dist={},
                        fidelity_plus=0.0, sparsity=0.0,
                        num_nodes=0, num_edges=0, error=str(e),
                    ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def append_to_csv(new_df: pd.DataFrame, csv_path: str):
    """Append new rows to the shared CSV (creates it if it doesn't exist)."""
    os.makedirs(os.path.dirname(csv_path) or "results", exist_ok=True)
    df_to_save = new_df.drop(columns=['jacobian_by_dist'], errors='ignore')
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        combined = pd.concat([existing, df_to_save], ignore_index=True)
    else:
        combined = df_to_save
    combined.to_csv(csv_path, index=False)
    print(f"[save] CSV → {csv_path}  ({len(combined)} total rows)")


def _update_json(new_df: pd.DataFrame, json_path: str):
    """Append new records to the shared JSON (preserves jacobian_by_dist dicts)."""
    records = new_df.copy()
    records['jacobian_by_dist'] = records['jacobian_by_dist'].apply(
        lambda d: {int(k): float(v) for k, v in d.items()}
        if isinstance(d, dict) else {}
    )
    new_records = records.to_dict(orient='records')

    if os.path.exists(json_path):
        with open(json_path) as f:
            existing_records = json.load(f)
        all_records = existing_records + new_records
    else:
        all_records = new_records

    with open(json_path, 'w') as f:
        json.dump(all_records, f, indent=2, default=str)
    print(f"[save] JSON → {json_path}  ({len(all_records)} total records)")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def generate_plots(csv_path: str, datasets: list):
    """Reload the full CSV and regenerate plots including signal strategies."""
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    for ds in datasets:
        ds_df = df[df['dataset'] == ds]
        if ds_df.empty:
            continue
        plot_tradeoff_curves(
            ds_df, dataset_name=ds,
            save_path=f"results/tradeoff_{ds}_signal.png",
        )
        plot_pareto_frontier(
            ds_df, dataset_name=ds,
            save_path=f"results/pareto_{ds}_signal.png",
        )
        print(f"  Plots → results/tradeoff_{ds}_signal.png, "
              f"results/pareto_{ds}_signal.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Signal-Guided Rewiring experiment (Ring + Cora)"
    )
    parser.add_argument('--datasets', nargs='+', default=['ring', 'cora'],
                        choices=['ring', 'cora'])
    parser.add_argument('--budgets', nargs='+', type=float, default=BUDGETS)
    parser.add_argument('--strategies', nargs='+', default=SIGNAL_STRATEGIES,
                        choices=SIGNAL_STRATEGIES)
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    print(f"[device] {get_device()}")

    df = run_signal_sweep(
        datasets=args.datasets,
        strategies=args.strategies,
        budgets=args.budgets,
        verbose=not args.quiet,
    )

    if df.empty:
        print("No results to save.")
        return

    append_to_csv(df, CSV_PATH)
    _update_json(df, JSON_PATH)
    generate_plots(CSV_PATH, args.datasets)

    print("\n" + "=" * 60)
    print("Signal-Guided Rewiring — Summary")
    print("=" * 60)
    summary_cols = ['dataset', 'strategy', 'budget', 'test_acc',
                    'jacobian_mean', 'mad', 'dirichlet']
    print(df[[c for c in summary_cols if c in df.columns]].to_string(index=False))


if __name__ == '__main__':
    main()
