"""
Visualization module for LRE-GAT over-squashing experiments.

Provides:
    plot_report()             — 4-panel summary figure (accuracy, Jacobian, fidelity, sparsity)
    plot_attention_heatmap()  — per-layer attention matrix as heatmap
    plot_attention_comparison() — side-by-side heatmaps across rewiring strategies
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import os
import torch

from models import attention_to_matrix


# ---------------------------------------------------------------------------
# Palette  (VirtualNode removed)
# ---------------------------------------------------------------------------

STRATEGY_ORDER  = ["Baseline", "Random", "K-hop", "FeatureSim"]
PALETTE         = sns.color_palette("husl", len(STRATEGY_ORDER))
STRATEGY_COLORS = dict(zip(STRATEGY_ORDER, PALETTE))


# ===========================================================================
# 1.  4-panel report
# ===========================================================================

def plot_report(results: dict, save_path: str = "results/report.png"):
    """
    results: dict  strategy_name → {
        test_acc, jacobian_by_dist, jacobian_mean, fidelity_plus, sparsity
    }
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    strategies = [s for s in STRATEGY_ORDER if s in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("LRE-GAT: Over-Squashing Analysis", fontsize=15, fontweight='bold')

    # --- 1. Accuracy Bar ------------------------------------------------- #
    ax = axes[0, 0]
    accs   = [results[s]['test_acc'] for s in strategies]
    colors = [STRATEGY_COLORS[s] for s in strategies]
    bars   = ax.bar(strategies, accs, color=colors, edgecolor='black', linewidth=0.7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("1. Accuracy by Rewiring Strategy")
    ax.tick_params(axis='x', rotation=15)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.3f}", ha='center', va='bottom', fontsize=9)

    # --- 2. Jacobian Norm vs Distance Heatmap ---------------------------- #
    ax = axes[0, 1]
    all_dists = sorted({d for s in strategies
                        for d in results[s].get('jacobian_by_dist', {}).keys()})
    heat_data = []
    for s in strategies:
        jbd = results[s].get('jacobian_by_dist', {})
        heat_data.append([jbd.get(d, 0.0) for d in all_dists])
    heat_arr = np.array(heat_data)
    if heat_arr.max() > 0:
        heat_arr = heat_arr / heat_arr.max()

    sns.heatmap(heat_arr, ax=ax, xticklabels=all_dists, yticklabels=strategies,
                cmap='YlOrRd', annot=True, fmt=".2f", cbar_kws={'label': 'Norm. Jacobian'})
    ax.set_xlabel("Hop Distance")
    ax.set_title("2. Jacobian Norm vs Hop Distance")

    # --- 3. Jacobian Mean vs Fidelity+ Scatter --------------------------- #
    ax = axes[1, 0]
    jac_vals = [results[s].get('jacobian_mean', 0.0) for s in strategies]
    fid_vals = [results[s].get('fidelity_plus', 0.0) for s in strategies]
    for s, jv, fv in zip(strategies, jac_vals, fid_vals):
        ax.scatter(jv, fv, s=120, color=STRATEGY_COLORS[s], label=s,
                   edgecolors='black', linewidths=0.6, zorder=3)
    if len(strategies) >= 2 and len(set(jac_vals)) > 1:
        z = np.polyfit(jac_vals, fid_vals, 1)
        xs = np.linspace(min(jac_vals), max(jac_vals), 100)
        ax.plot(xs, np.poly1d(z)(xs), 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel("Mean Jacobian Norm")
    ax.set_ylabel("Fidelity+")
    ax.set_title("3. Jacobian Norm vs Fidelity+")
    ax.legend(fontsize=8)

    # --- 4. Sparsity Bar ------------------------------------------------- #
    ax = axes[1, 1]
    sparsities = [results[s].get('sparsity', 0.0) for s in strategies]
    bars = ax.bar(strategies, sparsities, color=colors, edgecolor='black', linewidth=0.7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Explanation Sparsity (fraction edges masked)")
    ax.set_title("4. Explanation Sparsity per Strategy")
    ax.tick_params(axis='x', rotation=15)
    for bar, sp in zip(bars, sparsities):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{sp:.2f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[visualize] Report saved → {save_path}")


# ===========================================================================
# 2.  Single-strategy attention heatmap
# ===========================================================================

def plot_attention_heatmap(
    model,
    data,
    strategy_name: str = "unknown",
    layer_idx: int = -1,
    aggr: str = "mean",
    save_path: str = "results/attention_heatmap.png",
    highlight_artificial: bool = True,
    original_edge_index=None,
):
    """
    Extract attention weights from a trained GAT and display as a (N, N) heatmap.

    Args:
        model:                Trained GAT instance (will be set to eval mode).
        data:                 PyG Data object (single graph).
        strategy_name:        Label for the plot title.
        layer_idx:            Which layer's attention to visualise (-1 = last).
        aggr:                 Head aggregation — 'mean' or 'max'.
        save_path:            Output file path.
        highlight_artificial: If True, draw red rectangles over artificial edges.
        original_edge_index:  edge_index BEFORE rewiring — used to identify
                              which edges are artificial (added by rewiring).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model.eval()
    with torch.no_grad():
        _, attn_list = model(data.x, data.edge_index, return_attention_weights=True)

    # Select layer
    layer_idx = layer_idx % len(attn_list)
    ei_layer, alpha_layer = attn_list[layer_idx]   # (2,E), (E, H)

    num_nodes = data.num_nodes
    A = attention_to_matrix(ei_layer, alpha_layer, num_nodes, aggr=aggr)
    A_np = A.numpy()

    # ------------------------------------------------------------------ #
    # Build figure
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(max(8, num_nodes // 2), max(7, num_nodes // 2)))

    layer_label = f"Layer {layer_idx + 1} / {len(attn_list)}"
    im = ax.imshow(A_np, cmap='hot', aspect='auto', vmin=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"Attention weight αᵢⱼ  ({aggr} over {alpha_layer.shape[-1]} heads)",
                   fontsize=9)

    ax.set_title(
        f"Attention Heatmap — {strategy_name}  |  {layer_label}",
        fontsize=12, fontweight='bold'
    )
    ax.set_xlabel("Source node  j", fontsize=10)
    ax.set_ylabel("Target node  i", fontsize=10)

    ticks = np.arange(num_nodes)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks, fontsize=7)
    ax.set_yticklabels(ticks, fontsize=7)

    # ------------------------------------------------------------------ #
    # Highlight artificial edges (rewiring shortcuts)
    # ------------------------------------------------------------------ #
    patches = []
    if highlight_artificial and original_edge_index is not None:
        orig_set = set(map(tuple, original_edge_index.t().tolist()))
        new_set  = set(map(tuple, data.edge_index.t().tolist()))
        artificial = new_set - orig_set

        for (src, tgt) in artificial:
            # Rectangle: col=src (x-axis), row=tgt (y-axis)
            rect = mpatches.Rectangle(
                (src - 0.5, tgt - 0.5), 1, 1,
                linewidth=1.2, edgecolor='cyan', facecolor='none', zorder=5,
            )
            ax.add_patch(rect)

        if artificial:
            proxy = mpatches.Patch(edgecolor='cyan', facecolor='none', linewidth=1.5,
                                   label=f'Artificial edges  (n={len(artificial)})')
            patches.append(proxy)

    # Mark node 0 (signal source) and target node
    target = num_nodes // 2
    for node, color, lbl in [(0, 'lime', 'Signal source (node 0)'),
                              (target, 'deepskyblue', f'Target node ({target})')]:
        ax.axhline(node, color=color, linewidth=0.8, linestyle='--', alpha=0.7)
        ax.axvline(node, color=color, linewidth=0.8, linestyle='--', alpha=0.7)
        patches.append(mpatches.Patch(color=color, label=lbl))

    if patches:
        ax.legend(handles=patches, loc='upper right', fontsize=7,
                  framealpha=0.85, borderpad=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[visualize] Attention heatmap saved → {save_path}")


# ===========================================================================
# 3.  Multi-strategy comparison  (one column per strategy, one row per layer)
# ===========================================================================

def plot_attention_comparison(
    strategy_results: dict,
    save_path: str = "results/attention_comparison.png",
    aggr: str = "mean",
):
    """
    Side-by-side attention heatmaps for multiple rewiring strategies.

    Args:
        strategy_results: dict  strategy_name → {
            'model':               trained GAT,
            'data':                PyG Data (rewired graph),
            'original_edge_index': edge_index before rewiring (optional),
        }
        save_path: Output file path.
        aggr:      Head aggregation method ('mean' or 'max').
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    strategies = [s for s in STRATEGY_ORDER if s in strategy_results]
    if not strategies:
        print("[visualize] No strategies found — skipping comparison heatmap.")
        return

    # Collect attention matrices — shape (num_layers, N, N) per strategy
    all_matrices = {}
    num_layers_ref = None
    num_nodes_ref  = None

    for name in strategies:
        entry = strategy_results[name]
        model = entry['model']
        data  = entry['data']
        model.eval()

        with torch.no_grad():
            _, attn_list = model(data.x, data.edge_index, return_attention_weights=True)

        mats = []
        for ei_l, alpha_l in attn_list:
            A = attention_to_matrix(ei_l, alpha_l, data.num_nodes, aggr=aggr)
            mats.append(A.numpy())

        all_matrices[name] = mats
        num_layers_ref = len(mats)
        num_nodes_ref  = data.num_nodes

    n_rows = num_layers_ref
    n_cols = len(strategies)

    cell_size = max(3, num_nodes_ref // 5)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(cell_size * n_cols, cell_size * n_rows),
        squeeze=False,
    )
    fig.suptitle(
        f"Attention Weight Comparison  ({aggr} over heads)",
        fontsize=13, fontweight='bold', y=1.01,
    )

    for col_i, name in enumerate(strategies):
        mats = all_matrices[name]
        orig_ei = strategy_results[name].get('original_edge_index', None)
        data_ei = strategy_results[name]['data'].edge_index

        # Precompute artificial edge set once per strategy
        if orig_ei is not None:
            orig_set = set(map(tuple, orig_ei.t().tolist()))
            new_set  = set(map(tuple, data_ei.t().tolist()))
            artificial = new_set - orig_set
        else:
            artificial = set()

        for row_i, A_np in enumerate(mats):
            ax = axes[row_i][col_i]
            im = ax.imshow(A_np, cmap='hot', aspect='auto', vmin=0)

            if row_i == 0:
                ax.set_title(name, fontsize=10, fontweight='bold',
                             color=STRATEGY_COLORS.get(name, 'black'))
            if col_i == 0:
                ax.set_ylabel(f"Layer {row_i + 1}", fontsize=9)

            ax.set_xticks([])
            ax.set_yticks([])

            # Mark artificial edges
            for (src, tgt) in artificial:
                rect = mpatches.Rectangle(
                    (src - 0.5, tgt - 0.5), 1, 1,
                    linewidth=0.9, edgecolor='cyan', facecolor='none', zorder=5,
                )
                ax.add_patch(rect)

            # Mark signal source and target
            target = num_nodes_ref // 2
            ax.axhline(0,      color='lime',        linewidth=0.7, linestyle='--', alpha=0.8)
            ax.axvline(0,      color='lime',        linewidth=0.7, linestyle='--', alpha=0.8)
            ax.axhline(target, color='deepskyblue', linewidth=0.7, linestyle='--', alpha=0.8)
            ax.axvline(target, color='deepskyblue', linewidth=0.7, linestyle='--', alpha=0.8)

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[visualize] Attention comparison saved → {save_path}")


# ===========================================================================
# 4.  Per-layer attention stats (text summary helper)
# ===========================================================================

def summarize_attention_on_artificial(
    model,
    data,
    original_edge_index: torch.Tensor,
    strategy_name: str = "",
) -> dict:
    """
    For each GAT layer, compute mean attention weight on artificial vs original edges.

    Returns:
        dict layer_idx → {'artificial_mean': float, 'original_mean': float,
                          'ratio': float, 'n_artificial': int}
    """
    model.eval()
    with torch.no_grad():
        _, attn_list = model(data.x, data.edge_index, return_attention_weights=True)

    orig_set = set(map(tuple, original_edge_index.t().tolist()))
    ei_full  = data.edge_index

    summary = {}
    for layer_idx, (ei_l, alpha_l) in enumerate(attn_list):
        alpha_mean = alpha_l.mean(dim=-1).detach().cpu()  # (E,)

        art_weights, orig_weights = [], []
        for e_idx in range(ei_l.shape[1]):
            src, tgt = ei_l[0, e_idx].item(), ei_l[1, e_idx].item()
            w = alpha_mean[e_idx].item()
            if (src, tgt) in orig_set or (tgt, src) in orig_set:
                orig_weights.append(w)
            else:
                art_weights.append(w)

        art_mean  = float(np.mean(art_weights))  if art_weights  else 0.0
        orig_mean = float(np.mean(orig_weights)) if orig_weights else 1e-9
        ratio     = art_mean / orig_mean if orig_mean > 1e-12 else 0.0

        summary[layer_idx] = {
            'artificial_mean': art_mean,
            'original_mean':   orig_mean,
            'ratio':           ratio,
            'n_artificial':    len(art_weights),
        }

        tag = strategy_name or "?"
        print(f"  [{tag}] Layer {layer_idx + 1}: "
              f"art_α={art_mean:.4f}  orig_α={orig_mean:.4f}  "
              f"ratio={ratio:.2f}  (n_art={len(art_weights)})")

    return summary
