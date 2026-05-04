"""
Five graph rewiring strategies for LRE-GAT experiments.
All return a new edge_index (original + added edges), except VirtualNode
which also returns the modified node features and a flag.
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import (
    to_undirected, remove_self_loops,
)
from torch_geometric.utils import to_dense_adj
import numpy as np


def _budget_from_khop(edge_index: torch.Tensor, num_nodes: int, k: int) -> int:
    """Count how many new edges K-hop rewiring would add (used to set Random budget)."""
    # BFS/shortest-path to find all pairs at distance k
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0].numpy()
    # Floyd-Warshall is fine for small graphs (RingTransfer=20 nodes)
    dist = np.full((num_nodes, num_nodes), np.inf)
    np.fill_diagonal(dist, 0)
    rows, cols = np.where(adj > 0)
    dist[rows, cols] = 1
    for mid in range(num_nodes):
        dist = np.minimum(dist, dist[:, mid:mid+1] + dist[mid:mid+1, :])
    new_edges = np.sum(dist == k) // 2  # undirected pairs
    return int(new_edges)


def _pairwise_distances(edge_index: torch.Tensor, num_nodes: int) -> np.ndarray:
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0].numpy()
    dist = np.full((num_nodes, num_nodes), np.inf)
    np.fill_diagonal(dist, 0)
    rows, cols = np.where(adj > 0)
    dist[rows, cols] = 1
    for mid in range(num_nodes):
        dist = np.minimum(dist, dist[:, mid:mid+1] + dist[mid:mid+1, :])
    return dist


def baseline(data: Data) -> Data:
    """Return data unchanged."""
    return data


def random_rewiring(data: Data, k: int = 3, seed: int = 0) -> Data:
    """Add random edges with budget equal to K-hop rewiring."""
    torch.manual_seed(seed)
    num_nodes = data.num_nodes
    budget = _budget_from_khop(data.edge_index, num_nodes, k)
    if budget == 0:
        return data

    existing = set(map(tuple, data.edge_index.t().tolist()))
    new_edges = []
    attempts = 0
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

    new_t = torch.tensor(new_edges, dtype=torch.long).t()
    new_ei = to_undirected(new_t, num_nodes=num_nodes)
    combined = torch.cat([data.edge_index, new_ei], dim=1)
    combined, _ = remove_self_loops(combined)
    new_data = data.clone()
    new_data.edge_index = combined
    return new_data


def khop_rewiring(data: Data, k: int = 3) -> Data:
    """Add shortcuts between nodes at exact shortest-path distance k."""
    num_nodes = data.num_nodes
    dist = _pairwise_distances(data.edge_index, num_nodes)
    pairs = np.argwhere(dist == k)
    # Keep only upper triangle to avoid duplicates
    pairs = pairs[pairs[:, 0] < pairs[:, 1]]

    if len(pairs) == 0:
        return data

    new_t = torch.tensor(pairs, dtype=torch.long).t()
    new_ei = to_undirected(new_t, num_nodes=num_nodes)
    combined = torch.cat([data.edge_index, new_ei], dim=1)
    combined, _ = remove_self_loops(combined)
    new_data = data.clone()
    new_data.edge_index = combined
    return new_data


def feature_similarity_rewiring(data: Data, k: int = 3,
                                  train_mask: torch.Tensor = None) -> Data:
    """
    Add edges between nodes with highest cosine similarity.
    Budget = same as K-hop. Similarity computed only on train nodes
    to avoid data leakage (features of test nodes not used to select edges).
    """
    num_nodes = data.num_nodes
    budget = _budget_from_khop(data.edge_index, num_nodes, k)
    if budget == 0:
        return data

    x = data.x  # (N, F)
    x_norm = F.normalize(x, p=2, dim=1)
    sim = x_norm @ x_norm.t()  # (N, N)

    # Zero out diagonal and existing edges to avoid self-loops / duplicates
    sim.fill_diagonal_(-2.0)
    existing = data.edge_index
    sim[existing[0], existing[1]] = -2.0

    # If train_mask given, only use train nodes as sources to pick edges
    if train_mask is not None:
        train_idx = train_mask.nonzero(as_tuple=True)[0]
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[train_idx] = True
        sim[~mask, :] = -2.0

    # Pick top-budget pairs
    flat = sim.view(-1)
    top_vals, top_idx = flat.topk(budget * 2)  # *2 because directed → dedup
    rows = top_idx // num_nodes
    cols = top_idx % num_nodes
    valid = rows < cols  # upper triangle only
    rows, cols = rows[valid][:budget], cols[valid][:budget]

    if len(rows) == 0:
        return data

    new_t = torch.stack([rows, cols], dim=0)
    new_ei = to_undirected(new_t, num_nodes=num_nodes)
    combined = torch.cat([data.edge_index, new_ei], dim=1)
    combined, _ = remove_self_loops(combined)
    new_data = data.clone()
    new_data.edge_index = combined
    return new_data


def virtual_node_rewiring(data: Data) -> Data:
    """
    Add a virtual node connected to every real node.
    Virtual node features = mean of all real node features.
    Returns modified data with num_nodes+1 nodes.
    """
    num_nodes = data.num_nodes
    vn_idx = num_nodes  # index of virtual node

    # Edges: virtual node ↔ every real node
    real_nodes = torch.arange(num_nodes, dtype=torch.long)
    vn_nodes = torch.full((num_nodes,), vn_idx, dtype=torch.long)
    new_edges = torch.stack([
        torch.cat([real_nodes, vn_nodes]),
        torch.cat([vn_nodes, real_nodes]),
    ], dim=0)

    combined = torch.cat([data.edge_index, new_edges], dim=1)
    vn_feat = data.x.mean(dim=0, keepdim=True)  # (1, F)
    new_x = torch.cat([data.x, vn_feat], dim=0)  # (N+1, F)
    new_y = torch.cat([data.y, torch.tensor([-1])], dim=0)

    # Extend masks
    def extend_mask(mask):
        return torch.cat([mask, torch.zeros(1, dtype=torch.bool)])

    new_data = Data(
        x=new_x,
        edge_index=combined,
        y=new_y,
        train_mask=extend_mask(data.train_mask),
        val_mask=extend_mask(data.val_mask),
        test_mask=extend_mask(data.test_mask),
        num_classes=data.num_classes if hasattr(data, 'num_classes') else None,
        is_virtual_node=True,
        original_num_nodes=num_nodes,
    )
    return new_data


STRATEGIES = {
    "Baseline": baseline,
    "Random": random_rewiring,
    "K-hop": khop_rewiring,
    "FeatureSim": feature_similarity_rewiring,
    "VirtualNode": virtual_node_rewiring,
}
