"""
Signal-Guided Rewiring: uses internal signals from a pre-trained GAT to select
which edges to add, guided by either Jacobian sensitivity or accumulated
multi-hop attention flow.

Two strategies:
  JacobianGuided   — selects pairs with high ‖∂h_i/∂x_j‖, penalizing hubs.
  AttentionGuided  — selects pairs with high accumulated multi-hop attention,
                     penalizing hubs.

Both reuse helpers from rewiring.py:
  _shortest_paths, _existing_pair_set, _add_edges, _budget_edges
"""

import torch
import numpy as np
from torch_geometric.data import Data

from rewiring import (
    _shortest_paths,
    _existing_pair_set,
    _add_edges,
    _budget_edges,
)
from models import attention_to_matrix


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _sample_node_indices(num_nodes: int, max_nodes: int, seed: int) -> list:
    """Random subset of node indices without replacement (for large-graph Jacobian)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(num_nodes, generator=g)[:max_nodes].tolist()


def _select_with_hub_cap(
    scored_candidates: list,
    budget: int,
    num_nodes: int,
    mean_degree: float,
    hub_cap_factor: float = 2.0,
) -> list:
    """
    Greedy selection from scored_candidates (sorted descending by score),
    skipping pairs that would push either node's extra-degree above degree_cap.

    degree_cap = max(1, int(mean_degree * hub_cap_factor))
    """
    degree_cap   = max(1, int(mean_degree * hub_cap_factor))
    degree_added = np.zeros(num_nodes, dtype=np.int32)
    selected = []
    for _score, i, j in scored_candidates:
        if len(selected) >= budget:
            break
        if degree_added[i] >= degree_cap or degree_added[j] >= degree_cap:
            continue
        selected.append((i, j))
        degree_added[i] += 1
        degree_added[j] += 1
    return selected


# ---------------------------------------------------------------------------
# Jacobian-guided rewiring
# ---------------------------------------------------------------------------

def jacobian_guided_rewiring(
    model,
    data: Data,
    budget_frac: float = 0.25,
    min_hop: int = 2,
    hub_alpha: float = 0.5,
    hub_cap_factor: float = 2.0,
    seed: int = 0,
    graph_level: bool = False,
    max_nodes_jacobian: int = 200,
) -> Data:
    """
    Rewires `data` by adding edges between nodes with high Jacobian sensitivity,
    discounted by a hub penalty to avoid degree concentration.

    Node-level (Ring, Cora):
        Computes the N×N matrix J[i,j] = ‖∂logits_i/∂x_j‖ for each source node i.
        For large graphs (N > max_nodes_jacobian) only a random subset of rows are
        computed — unsampled rows default to 0 and are not selected.

    Graph-level:
        One backward pass over logits.sum() gives importance[j] = ‖∂L/∂x_j‖.
        Symmetrized as J[i,j] = importance[i] + importance[j].

    Scoring:  score(i,j) = J[i,j] / (deg_i+1 * deg_j+1)^hub_alpha
    Filters:  dist(i,j) >= min_hop, (i,j) not already connected.
    """
    if budget_frac <= 0:
        return data
    num_nodes = data.num_nodes
    if num_nodes < 3:
        return data
    budget = _budget_edges(data.edge_index, budget_frac)
    if budget == 0:
        return data

    device   = data.x.device
    dist     = _shortest_paths(data.edge_index.cpu(), num_nodes)
    existing = _existing_pair_set(data.edge_index.cpu())

    model.eval()

    # ---- Compute Jacobian importance matrix J (N, N) ----
    if graph_level:
        x_in  = data.x.clone().detach().requires_grad_(True)
        batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
        logits = model(x_in, data.edge_index, batch)
        logits.sum().backward()
        importance = x_in.grad.norm(dim=1).detach().cpu().numpy()  # (N,)
        J = importance[:, None] + importance[None, :]              # (N, N)
    else:
        if num_nodes > max_nodes_jacobian:
            source_nodes = _sample_node_indices(num_nodes, max_nodes_jacobian, seed)
        else:
            source_nodes = list(range(num_nodes))

        J = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for i in source_nodes:
            x_in   = data.x.clone().detach().requires_grad_(True)
            logits = model(x_in, data.edge_index)
            logits[i].sum().backward()
            norms  = x_in.grad.norm(dim=1).detach().cpu().numpy()
            J[i, :] = norms

    # ---- Hub penalty ----
    src_np      = data.edge_index[0].cpu().numpy()
    deg         = np.bincount(src_np, minlength=num_nodes).astype(np.float32)
    mean_degree = float(deg.mean()) if deg.mean() > 0 else 1.0
    hub_penalty = np.outer(deg + 1.0, deg + 1.0) ** hub_alpha  # (N, N)

    # ---- Scored candidates ----
    candidates = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if (i, j) in existing:
                continue
            d = dist[i, j]
            if np.isinf(d) or d < min_hop:
                continue
            score = float(J[i, j]) / hub_penalty[i, j]
            candidates.append((score, i, j))

    if not candidates:
        return data

    candidates.sort(key=lambda t: t[0], reverse=True)
    new_pairs = _select_with_hub_cap(
        candidates, budget, num_nodes, mean_degree, hub_cap_factor
    )
    return _add_edges(data, new_pairs, num_nodes)


# ---------------------------------------------------------------------------
# Attention-guided rewiring
# ---------------------------------------------------------------------------

def attention_guided_rewiring(
    model,
    data: Data,
    budget_frac: float = 0.25,
    min_hop: int = 2,
    num_hops: int = None,
    hub_alpha: float = 0.5,
    hub_cap_factor: float = 2.0,
    aggr: str = "mean",
    seed: int = 0,
    graph_level: bool = False,
) -> Data:
    """
    Rewires `data` using accumulated multi-hop attention as a proxy for
    information flow. Selects distant pairs with high attention mass,
    discounted by a hub penalty.

    Attention accumulation:
        A_accum = A_L @ A_{L-1} @ ... @ A_1  (matrix product, left-to-right)
        Column-normalized so scores are comparable across graphs of different density.

    Scoring:  score(i,j) = max(A_accum[i,j], A_accum[j,i]) / (deg_i+1 * deg_j+1)^hub_alpha
    Filters:  dist(i,j) >= min_hop, (i,j) not already connected.

    num_hops: if not None, uses only the last num_hops layers for accumulation.
    """
    if budget_frac <= 0:
        return data
    num_nodes = data.num_nodes
    if num_nodes < 3:
        return data
    budget = _budget_edges(data.edge_index, budget_frac)
    if budget == 0:
        return data

    device = data.x.device

    # ---- Extract per-layer attention matrices ----
    model.eval()
    with torch.no_grad():
        if graph_level:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
            _, attention_list = model(
                data.x, data.edge_index, batch,
                return_attention_weights=True,
            )
        else:
            _, attention_list = model(
                data.x, data.edge_index,
                return_attention_weights=True,
            )

    if not attention_list:
        return data

    A_layers = []
    for (ei_l, alpha_l) in attention_list:
        A_l = attention_to_matrix(ei_l, alpha_l.detach(), num_nodes, aggr=aggr)
        A_layers.append(A_l)  # (N, N) float tensor

    if num_hops is not None and num_hops > 0:
        A_layers = A_layers[-num_hops:]

    # ---- Accumulate: A_L @ A_{L-1} @ ... @ A_1 ----
    A_accum = A_layers[0]
    for A_l in A_layers[1:]:
        A_accum = A_l @ A_accum

    col_sums = A_accum.sum(dim=0, keepdim=True).clamp(min=1e-8)
    A_np = (A_accum / col_sums).cpu().numpy()  # (N, N)

    # ---- Hub penalty and distances ----
    dist     = _shortest_paths(data.edge_index.cpu(), num_nodes)
    existing = _existing_pair_set(data.edge_index.cpu())

    src_np      = data.edge_index[0].cpu().numpy()
    deg         = np.bincount(src_np, minlength=num_nodes).astype(np.float32)
    mean_degree = float(deg.mean()) if deg.mean() > 0 else 1.0
    hub_penalty = np.outer(deg + 1.0, deg + 1.0) ** hub_alpha

    # ---- Scored candidates ----
    candidates = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if (i, j) in existing:
                continue
            d = dist[i, j]
            if np.isinf(d) or d < min_hop:
                continue
            score = float(max(A_np[i, j], A_np[j, i])) / hub_penalty[i, j]
            candidates.append((score, i, j))

    if not candidates:
        return data

    candidates.sort(key=lambda t: t[0], reverse=True)
    new_pairs = _select_with_hub_cap(
        candidates, budget, num_nodes, mean_degree, hub_cap_factor
    )
    return _add_edges(data, new_pairs, num_nodes)
