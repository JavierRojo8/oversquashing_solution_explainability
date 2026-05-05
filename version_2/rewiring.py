"""
Graph rewiring strategies for LRE-GAT experiments.

Four strategies — all operate exclusively on existing nodes.
No virtual nodes. No changes to num_nodes.
All return a new Data object with edge_index = original + added edges.
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, remove_self_loops, to_dense_adj
import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pairwise_distances(edge_index: torch.Tensor, num_nodes: int) -> np.ndarray:
    """Floyd-Warshall shortest-path distances. Fine for small graphs (≤ ~200 nodes)."""
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0].numpy()
    dist = np.full((num_nodes, num_nodes), np.inf)
    np.fill_diagonal(dist, 0)
    rows, cols = np.where(adj > 0)
    dist[rows, cols] = 1.0
    for mid in range(num_nodes):
        dist = np.minimum(dist, dist[:, mid:mid + 1] + dist[mid:mid + 1, :])
    return dist


def _budget_from_khop(edge_index: torch.Tensor, num_nodes: int, k: int) -> int:
    """Count undirected pairs at exact distance k (used to set budget for Random/FeatureSim)."""
    dist = _pairwise_distances(edge_index, num_nodes)
    pairs = np.sum(dist == k) // 2      # upper-triangle count
    return int(pairs)


# ---------------------------------------------------------------------------
# Strategy: Baseline
# ---------------------------------------------------------------------------

def baseline(data: Data) -> Data:
    """Return data unchanged — no rewiring."""
    return data


# ---------------------------------------------------------------------------
# Strategy: Random
# ---------------------------------------------------------------------------

def random_rewiring(data: Data, k: int = 3, seed: int = 0) -> Data:
    """
    Add randomly chosen long-range edges.
    Budget = number of edges K-hop rewiring would add (fair comparison).
    """
    torch.manual_seed(seed)
    num_nodes = data.num_nodes
    budget = _budget_from_khop(data.edge_index, num_nodes, k)
    if budget == 0:
        return data

    existing = set(map(tuple, data.edge_index.t().tolist()))
    new_edges, attempts = [], 0
    while len(new_edges) < budget and attempts < budget * 100:
        i = torch.randint(0, num_nodes, (1,)).item()
        j = torch.randint(0, num_nodes, (1,)).item()
        if i != j and (i, j) not in existing and (j, i) not in existing:
            new_edges.append((i, j))
            existing.add((i, j))
            existing.add((j, i))
        attempts += 1

    if not new_edges:
        return data

    new_t  = torch.tensor(new_edges, dtype=torch.long).t()
    new_ei = to_undirected(new_t, num_nodes=num_nodes)
    combined, _ = remove_self_loops(torch.cat([data.edge_index, new_ei], dim=1))
    new_data = data.clone()
    new_data.edge_index = combined
    return new_data


# ---------------------------------------------------------------------------
# Strategy: K-hop
# ---------------------------------------------------------------------------

def khop_rewiring(data: Data, k: int = 3) -> Data:
    """
    Add shortcuts between every pair of nodes at exact shortest-path distance k.
    This is the primary structural rewiring strategy for over-squashing mitigation.
    """
    num_nodes = data.num_nodes
    dist = _pairwise_distances(data.edge_index, num_nodes)
    pairs = np.argwhere(dist == k)
    pairs = pairs[pairs[:, 0] < pairs[:, 1]]   # upper triangle only

    if len(pairs) == 0:
        return data

    new_t  = torch.tensor(pairs, dtype=torch.long).t()
    new_ei = to_undirected(new_t, num_nodes=num_nodes)
    combined, _ = remove_self_loops(torch.cat([data.edge_index, new_ei], dim=1))
    new_data = data.clone()
    new_data.edge_index = combined
    return new_data


# ---------------------------------------------------------------------------
# Strategy: Feature Similarity
# ---------------------------------------------------------------------------

def feature_similarity_rewiring(data: Data, k: int = 3,
                                  train_mask: torch.Tensor = None) -> Data:
    """
    Add edges between the top-budget pairs ranked by cosine similarity.
    Budget = same as K-hop (fair comparison).

    NOTE on train_mask: In RingTransfer each graph has exactly one active mask node.
    Using train_mask as a source filter collapses the similarity matrix to a single row,
    which is often too restrictive. When the active-node count < 2, the mask is ignored
    and similarity is computed globally (all nodes as potential sources).
    """
    num_nodes = data.num_nodes
    budget = _budget_from_khop(data.edge_index, num_nodes, k)
    if budget == 0:
        return data

    x_norm = F.normalize(data.x, p=2, dim=1)   # (N, F)
    sim = x_norm @ x_norm.t()                   # (N, N)

    # Mask out diagonal and already-existing edges
    sim.fill_diagonal_(-2.0)
    sim[data.edge_index[0], data.edge_index[1]] = -2.0

    # Apply train_mask only when it provides ≥ 2 candidate sources
    if train_mask is not None:
        train_idx = train_mask.nonzero(as_tuple=True)[0]
        if len(train_idx) >= 2:
            mask = torch.zeros(num_nodes, dtype=torch.bool)
            mask[train_idx] = True
            sim[~mask, :] = -2.0

    # Pick top-budget undirected pairs from upper triangle
    flat = sim.view(-1)
    top_idx = flat.topk(min(budget * 4, flat.numel())).indices
    rows = top_idx // num_nodes
    cols = top_idx % num_nodes
    upper = rows < cols
    rows, cols = rows[upper][:budget], cols[upper][:budget]

    if len(rows) == 0:
        return data

    new_t  = torch.stack([rows, cols], dim=0)
    new_ei = to_undirected(new_t, num_nodes=num_nodes)
    combined, _ = remove_self_loops(torch.cat([data.edge_index, new_ei], dim=1))
    new_data = data.clone()
    new_data.edge_index = combined
    return new_data


# ---------------------------------------------------------------------------
# Public registry  (VirtualNode removed)
# ---------------------------------------------------------------------------

STRATEGIES = {
    "Baseline":   baseline,
    "Random":     random_rewiring,
    "K-hop":      khop_rewiring,
    "FeatureSim": feature_similarity_rewiring,
}
