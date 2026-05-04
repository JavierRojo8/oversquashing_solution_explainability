"""
4-panel report figure for LRE-GAT experiment.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


STRATEGY_ORDER = ["Baseline", "Random", "K-hop", "FeatureSim", "VirtualNode"]
PALETTE = sns.color_palette("husl", len(STRATEGY_ORDER))
STRATEGY_COLORS = dict(zip(STRATEGY_ORDER, PALETTE))


def plot_report(results: dict, save_path: str = "results/report.png"):
    """
    results: dict with strategy names as keys. Each value is a dict with:
        - test_acc: float
        - jacobian_by_dist: {distance: mean_norm}
        - fidelity_plus: float (mean over nodes)
        - sparsity: float
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    strategies = [s for s in STRATEGY_ORDER if s in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("LRE-GAT: Over-Squashing Analysis", fontsize=15, fontweight='bold')

    # ------------------------------------------------------------------ #
    # 1. Accuracy Bar Chart
    # ------------------------------------------------------------------ #
    ax = axes[0, 0]
    accs = [results[s]['test_acc'] for s in strategies]
    colors = [STRATEGY_COLORS[s] for s in strategies]
    bars = ax.bar(strategies, accs, color=colors, edgecolor='black', linewidth=0.7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("1. Accuracy by Rewiring Strategy")
    ax.tick_params(axis='x', rotation=15)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.3f}", ha='center', va='bottom', fontsize=9)

    # ------------------------------------------------------------------ #
    # 2. Heatmap: Jacobian Norm vs Distance
    # ------------------------------------------------------------------ #
    ax = axes[0, 1]
    # Build matrix: rows=strategies, cols=distances
    all_dists = sorted(set(
        d for s in strategies
        for d in results[s].get('jacobian_by_dist', {}).keys()
    ))
    heat_data = []
    for s in strategies:
        jbd = results[s].get('jacobian_by_dist', {})
        row = [jbd.get(d, 0.0) for d in all_dists]
        heat_data.append(row)
    heat_arr = np.array(heat_data)
    if heat_arr.max() > 0:
        heat_arr = heat_arr / heat_arr.max()  # normalize for readability

    sns.heatmap(heat_arr, ax=ax, xticklabels=all_dists, yticklabels=strategies,
                cmap='YlOrRd', annot=True, fmt=".2f", cbar_kws={'label': 'Norm. Jacobian'})
    ax.set_xlabel("Hop Distance")
    ax.set_title("2. Jacobian Norm vs Hop Distance")

    # ------------------------------------------------------------------ #
    # 3. Correlation: Jacobian Norm vs Fidelity+
    # ------------------------------------------------------------------ #
    ax = axes[1, 0]
    jac_vals = [results[s].get('jacobian_mean', 0.0) for s in strategies]
    fid_vals = [results[s].get('fidelity_plus', 0.0) for s in strategies]
    for s, jv, fv in zip(strategies, jac_vals, fid_vals):
        ax.scatter(jv, fv, s=100, color=STRATEGY_COLORS[s], label=s,
                   edgecolors='black', linewidths=0.5, zorder=3)
    if len(strategies) >= 2 and any(jv != jac_vals[0] for jv in jac_vals):
        z = np.polyfit(jac_vals, fid_vals, 1)
        p = np.poly1d(z)
        xs = np.linspace(min(jac_vals), max(jac_vals), 100)
        ax.plot(xs, p(xs), 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel("Mean Jacobian Norm")
    ax.set_ylabel("Fidelity+")
    ax.set_title("3. Jacobian Norm vs Fidelity+")
    ax.legend(fontsize=8)

    # ------------------------------------------------------------------ #
    # 4. Sparsity Plot
    # ------------------------------------------------------------------ #
    ax = axes[1, 1]
    sparsities = [results[s].get('sparsity', 0.0) for s in strategies]
    colors = [STRATEGY_COLORS[s] for s in strategies]
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
    print(f"[visualize] Report saved to {save_path}")
