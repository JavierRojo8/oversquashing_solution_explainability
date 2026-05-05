"""
Visualization for the squashing-smoothing trade-off study.

Funciones principales:
  - plot_tradeoff_curves(): para un dataset, panel 4-subplot con
        budget→accuracy, budget→MAD, budget→Jacobian, accuracy↔MAD scatter.
  - plot_pareto_frontier(): scatter (smoothing, squashing) coloreado por accuracy.
  - plot_attention_heatmap(): heredado, sin cambios.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import os
import torch
import pandas as pd

from models import attention_to_matrix


# ---------------------------------------------------------------------------
# Estilo
# ---------------------------------------------------------------------------

STRATEGY_ORDER  = ["Baseline", "Random", "K-hop", "FeatureSim"]
PALETTE         = sns.color_palette("husl", len(STRATEGY_ORDER))
STRATEGY_COLORS = dict(zip(STRATEGY_ORDER, PALETTE))
STRATEGY_MARKERS = {"Baseline": "o", "Random": "s", "K-hop": "^", "FeatureSim": "D"}


# ===========================================================================
# 1. Trade-off curves: 4-panel por dataset
# ===========================================================================

def plot_tradeoff_curves(df: pd.DataFrame, dataset_name: str,
                          save_path: str = "results/tradeoff.png"):
    """
    df columnas requeridas:
        dataset, strategy, budget, test_acc, mad, dirichlet, jacobian_mean,
        fidelity_plus (opcional), sparsity (opcional)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sub = df[df['dataset'] == dataset_name].copy()
    if sub.empty:
        print(f"[viz] No data for {dataset_name}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f"Squashing↔Smoothing Trade-off — {dataset_name}",
                 fontsize=14, fontweight='bold')

    # --- (1) Budget → Accuracy ----------------------------------------- #
    ax = axes[0, 0]
    for s in STRATEGY_ORDER:
        rows = sub[sub['strategy'] == s].sort_values('budget')
        if rows.empty:
            continue
        ax.plot(rows['budget'], rows['test_acc'],
                marker=STRATEGY_MARKERS[s], color=STRATEGY_COLORS[s],
                label=s, linewidth=1.6, markersize=8)
    ax.set_xlabel("Budget (fracción de aristas extra)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy vs budget")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # --- (2) Budget → MAD (over-smoothing) ----------------------------- #
    ax = axes[0, 1]
    for s in STRATEGY_ORDER:
        rows = sub[sub['strategy'] == s].sort_values('budget')
        if rows.empty:
            continue
        ax.plot(rows['budget'], rows['mad'],
                marker=STRATEGY_MARKERS[s], color=STRATEGY_COLORS[s],
                label=s, linewidth=1.6, markersize=8)
    ax.set_xlabel("Budget (fracción de aristas extra)")
    ax.set_ylabel("MAD (cosine distance)")
    ax.set_title("Over-smoothing: MAD vs budget\n(bajo = smoothing alto)")
    ax.grid(alpha=0.3)

    # --- (3) Budget → Jacobian (over-squashing) ------------------------ #
    ax = axes[1, 0]
    for s in STRATEGY_ORDER:
        rows = sub[sub['strategy'] == s].sort_values('budget')
        if rows.empty:
            continue
        ax.plot(rows['budget'], rows['jacobian_mean'],
                marker=STRATEGY_MARKERS[s], color=STRATEGY_COLORS[s],
                label=s, linewidth=1.6, markersize=8)
    ax.set_xlabel("Budget (fracción de aristas extra)")
    ax.set_ylabel("Mean Jacobian norm")
    ax.set_title("Over-squashing: Jacobian vs budget\n(alto = info fluye)")
    ax.grid(alpha=0.3)

    # --- (4) Pareto: MAD vs Acc ---------------------------------------- #
    ax = axes[1, 1]
    for s in STRATEGY_ORDER:
        rows = sub[sub['strategy'] == s]
        if rows.empty:
            continue
        sc = ax.scatter(rows['mad'], rows['test_acc'],
                        c=rows['budget'], cmap='viridis',
                        marker=STRATEGY_MARKERS[s], s=80,
                        edgecolors=STRATEGY_COLORS[s], linewidths=1.5,
                        label=s)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Budget", fontsize=9)
    ax.set_xlabel("MAD (over-smoothing axis)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Trade-off: smoothing axis vs accuracy")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[viz] Trade-off saved → {save_path}")


# ===========================================================================
# 2. Pareto frontier squashing vs smoothing
# ===========================================================================

def plot_pareto_frontier(df: pd.DataFrame, dataset_name: str,
                          save_path: str = "results/pareto.png"):
    """
    Scatter (Jacobian, MAD) coloreado por accuracy. Muestra la frontera
    de Pareto y dónde cae cada estrategia/budget.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sub = df[df['dataset'] == dataset_name].copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    fig.suptitle(f"Pareto frontier — {dataset_name}", fontsize=13, fontweight='bold')

    sc = None
    for s in STRATEGY_ORDER:
        rows = sub[sub['strategy'] == s]
        if rows.empty:
            continue
        sc = ax.scatter(rows['jacobian_mean'], rows['mad'],
                        c=rows['test_acc'], cmap='RdYlGn',
                        marker=STRATEGY_MARKERS[s], s=110,
                        edgecolors=STRATEGY_COLORS[s], linewidths=1.5,
                        vmin=sub['test_acc'].min(),
                        vmax=sub['test_acc'].max())
        # Etiquetas con el budget
        for _, r in rows.iterrows():
            ax.annotate(f"{int(r['budget']*100)}%",
                        (r['jacobian_mean'], r['mad']),
                        fontsize=7, alpha=0.7,
                        xytext=(4, 4), textcoords='offset points')
    if sc is not None:
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label("Test accuracy")

    ax.set_xlabel("Jacobian mean (HIGH = no squashing)")
    ax.set_ylabel("MAD (HIGH = no smoothing)")
    ax.set_title("Top-right corner = best regime", fontsize=10)

    handles = [mpatches.Patch(color=STRATEGY_COLORS[s], label=s)
               for s in STRATEGY_ORDER if s in sub['strategy'].unique()]
    ax.legend(handles=handles, fontsize=9, loc='best')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[viz] Pareto saved → {save_path}")


# ===========================================================================
# 3. Cross-dataset summary
# ===========================================================================

def plot_cross_dataset_summary(df: pd.DataFrame,
                                save_path: str = "results/summary.png"):
    """
    Para cada dataset, mejor accuracy de cada estrategia (max sobre budgets).
    Gráfico de barras agrupadas para resumen final.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if df.empty:
        return

    best = df.groupby(['dataset', 'strategy'])['test_acc'].max().reset_index()
    pivot = best.pivot(index='dataset', columns='strategy', values='test_acc')
    pivot = pivot.reindex(columns=[s for s in STRATEGY_ORDER if s in pivot.columns])

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind='bar', ax=ax,
               color=[STRATEGY_COLORS[s] for s in pivot.columns],
               edgecolor='black', linewidth=0.6, width=0.78)
    ax.set_ylabel("Best test accuracy (max over budgets)")
    ax.set_title("Best accuracy per dataset × strategy",
                 fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, title="Strategy")
    ax.tick_params(axis='x', rotation=0)
    ax.grid(axis='y', alpha=0.3)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=7, padding=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[viz] Summary saved → {save_path}")


# ===========================================================================
# 4. Attention heatmap (heredado, simplificado para llamada one-shot)
# ===========================================================================

def plot_attention_heatmap(model, data, strategy_name: str = "unknown",
                            layer_idx: int = -1, aggr: str = "mean",
                            save_path: str = "results/attention.png",
                            highlight_artificial: bool = True,
                            original_edge_index=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.eval()
    with torch.no_grad():
        _, attn_list = model(data.x, data.edge_index, return_attention_weights=True)

    layer_idx = layer_idx % len(attn_list)
    ei_layer, alpha_layer = attn_list[layer_idx]
    num_nodes = data.num_nodes
    A = attention_to_matrix(ei_layer, alpha_layer, num_nodes, aggr=aggr).numpy()

    fig, ax = plt.subplots(figsize=(max(7, num_nodes // 3), max(6, num_nodes // 3)))
    im = ax.imshow(A, cmap='hot', aspect='auto', vmin=0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Attention — {strategy_name} (Layer {layer_idx + 1}/{len(attn_list)})",
                 fontsize=11, fontweight='bold')
    ax.set_xlabel("Source j"); ax.set_ylabel("Target i")

    if highlight_artificial and original_edge_index is not None:
        orig_set = set(map(tuple, original_edge_index.t().tolist()))
        new_set  = set(map(tuple, data.edge_index.t().tolist()))
        artificial = new_set - orig_set
        for (src, tgt) in artificial:
            ax.add_patch(mpatches.Rectangle(
                (src - 0.5, tgt - 0.5), 1, 1,
                linewidth=1.0, edgecolor='cyan', facecolor='none'))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[viz] Attention saved → {save_path}")
