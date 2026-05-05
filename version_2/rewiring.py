"""
Graph rewiring strategies — parametrizadas por BUDGET (fracción de aristas a añadir).

Esta versión sustituye el parámetro `k` fijo por `budget_frac`, que expresa
el número de aristas extras a añadir como fracción del número de aristas
del grafo original. Esto permite barrer el trade-off squashing↔smoothing:
budget bajo → poca densificación, budget alto → grafo casi denso.

Estrategias:
  - baseline:                 sin cambios.
  - random_rewiring:          aristas aleatorias.
  - khop_rewiring:            aristas entre nodos a distancia >= 2,
                              priorizando los más lejanos hasta consumir budget.
  - feature_similarity_rewiring: aristas entre los pares no conectados
                              con mayor similitud coseno, hasta consumir budget.

Todas devuelven un nuevo Data con num_nodes invariante.
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, remove_self_loops, to_dense_adj
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pairwise_distances(edge_index: torch.Tensor, num_nodes: int) -> np.ndarray:
    """Floyd-Warshall O(N^3). Para N <= ~300."""
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0].numpy()
    dist = np.full((num_nodes, num_nodes), np.inf)
    np.fill_diagonal(dist, 0)
    rows, cols = np.where(adj > 0)
    dist[rows, cols] = 1.0
    for mid in range(num_nodes):
        dist = np.minimum(dist, dist[:, mid:mid + 1] + dist[mid:mid + 1, :])
    return dist


def _bfs_distances(edge_index: torch.Tensor, num_nodes: int) -> np.ndarray:
    """BFS desde cada nodo. O(N*(N+E)) — más rápido que FW para grafos sparse."""
    from collections import deque
    adj = [[] for _ in range(num_nodes)]
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for s, d in zip(src, dst):
        adj[s].append(d)

    dist = np.full((num_nodes, num_nodes), np.inf)
    for start in range(num_nodes):
        dist[start, start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if np.isinf(dist[start, v]):
                    dist[start, v] = dist[start, u] + 1
                    q.append(v)
    return dist


def _shortest_paths(edge_index: torch.Tensor, num_nodes: int) -> np.ndarray:
    """Auto-elige BFS para grafos grandes, FW para pequeños."""
    if num_nodes > 100:
        return _bfs_distances(edge_index, num_nodes)
    return _pairwise_distances(edge_index, num_nodes)


def _budget_edges(edge_index: torch.Tensor, budget_frac: float) -> int:
    """budget_frac → nº entero de aristas (no-dirigidas) a añadir."""
    num_undirected = edge_index.shape[1] // 2
    return int(num_undirected * budget_frac)


def _existing_pair_set(edge_index: torch.Tensor) -> set:
    """Set de pares (i,j) con i<j ya presentes (no-dirigido)."""
    s = set()
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for a, b in zip(src, dst):
        if a == b:
            continue
        s.add((min(a, b), max(a, b)))
    return s


def _add_edges(data: Data, new_pairs: list, num_nodes: int) -> Data:
    """Devuelve nuevo Data con `new_pairs` añadidos (no-dirigido)."""
    if len(new_pairs) == 0:
        return data
    new_t  = torch.tensor(new_pairs, dtype=torch.long).t()
    new_ei = to_undirected(new_t, num_nodes=num_nodes)
    combined, _ = remove_self_loops(torch.cat([data.edge_index, new_ei], dim=1))
    new_data = data.clone()
    new_data.edge_index = combined
    return new_data


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def baseline(data: Data, **_) -> Data:
    return data


# ---------------------------------------------------------------------------
# Random
# ---------------------------------------------------------------------------

def random_rewiring(data: Data, budget_frac: float = 0.25, seed: int = 0) -> Data:
    """Aristas aleatorias entre pares no conectados."""
    if budget_frac <= 0:
        return data
    torch.manual_seed(seed)
    num_nodes = data.num_nodes
    budget = _budget_edges(data.edge_index, budget_frac)
    if budget == 0:
        return data

    existing = _existing_pair_set(data.edge_index)
    new_pairs, attempts = [], 0
    max_attempts = budget * 100
    while len(new_pairs) < budget and attempts < max_attempts:
        i = torch.randint(0, num_nodes, (1,)).item()
        j = torch.randint(0, num_nodes, (1,)).item()
        if i == j:
            attempts += 1
            continue
        a, b = (i, j) if i < j else (j, i)
        if (a, b) not in existing:
            new_pairs.append((a, b))
            existing.add((a, b))
        attempts += 1
    return _add_edges(data, new_pairs, num_nodes)


# ---------------------------------------------------------------------------
# K-hop  (versión budget: top pares por distancia decreciente)
# ---------------------------------------------------------------------------

def khop_rewiring(data: Data, budget_frac: float = 0.25, seed: int = 0) -> Data:
    """
    Añade aristas entre pares no conectados, priorizando los MÁS LEJANOS
    en el grafo original. Para budget pequeño conecta los puntos más distantes
    (donde el over-squashing es peor); para budget grande densifica el grafo.
    """
    if budget_frac <= 0:
        return data
    num_nodes = data.num_nodes
    budget = _budget_edges(data.edge_index, budget_frac)
    if budget == 0:
        return data

    dist = _shortest_paths(data.edge_index, num_nodes)
    existing = _existing_pair_set(data.edge_index)

    rng = np.random.default_rng(seed)
    candidates = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if (i, j) in existing:
                continue
            d = dist[i, j]
            if np.isinf(d) or d < 2:
                continue
            candidates.append((d, rng.random(), i, j))
    if not candidates:
        return data

    candidates.sort(key=lambda t: (-t[0], t[1]))
    selected = candidates[:budget]
    new_pairs = [(i, j) for (_, _, i, j) in selected]
    return _add_edges(data, new_pairs, num_nodes)


# ---------------------------------------------------------------------------
# Feature Similarity
# ---------------------------------------------------------------------------

def feature_similarity_rewiring(data: Data, budget_frac: float = 0.25,
                                 train_mask: torch.Tensor = None,
                                 seed: int = 0) -> Data:
    """Aristas entre pares no conectados con mayor similitud coseno."""
    if budget_frac <= 0:
        return data
    num_nodes = data.num_nodes
    budget = _budget_edges(data.edge_index, budget_frac)
    if budget == 0:
        return data

    x_norm = F.normalize(data.x, p=2, dim=1)
    sim = (x_norm @ x_norm.t()).cpu().numpy()

    np.fill_diagonal(sim, -2.0)
    sim[data.edge_index[0].cpu().numpy(),
        data.edge_index[1].cpu().numpy()] = -2.0

    if train_mask is not None:
        train_idx = train_mask.nonzero(as_tuple=True)[0].cpu().numpy()
        if len(train_idx) >= 2:
            mask = np.zeros(num_nodes, dtype=bool)
            mask[train_idx] = True
            sim[~mask, :] = -2.0

    iu, ju = np.triu_indices(num_nodes, k=1)
    sims_upper = sim[iu, ju]
    order = np.argsort(-sims_upper)

    new_pairs = []
    for idx in order:
        if sims_upper[idx] <= -1.5:
            continue
        new_pairs.append((int(iu[idx]), int(ju[idx])))
        if len(new_pairs) >= budget:
            break
    return _add_edges(data, new_pairs, num_nodes)


# ---------------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------------

STRATEGIES = ["Baseline", "Random", "K-hop", "FeatureSim"]


def apply_strategy(name: str, data: Data, budget_frac: float, seed: int = 0,
                   train_mask: torch.Tensor = None) -> Data:
    """Wrapper único para aplicar cualquier estrategia con budget."""
    if name == "Baseline":
        return baseline(data)
    if name == "Random":
        return random_rewiring(data, budget_frac=budget_frac, seed=seed)
    if name == "K-hop":
        return khop_rewiring(data, budget_frac=budget_frac, seed=seed)
    if name == "FeatureSim":
        return feature_similarity_rewiring(data, budget_frac=budget_frac,
                                            seed=seed, train_mask=train_mask)
    raise ValueError(f"Unknown strategy: {name}")
