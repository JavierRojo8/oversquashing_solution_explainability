"""
Métricas de diagnóstico para el trade-off over-squashing ↔ over-smoothing.

Proxies de OVER-SQUASHING:
  - jacobian_norm_by_distance(): ‖∂h_i^L/∂x_j‖ agrupado por distancia hop.
                                 Decay exponencial = squashing.

Proxies de OVER-SMOOTHING:
  - mean_average_distance() (MAD): distancia coseno media entre embeddings.
                                   Bajo = embeddings indistinguibles = smoothing.
  - dirichlet_energy(): ∑_{(i,j)∈E} ‖h_i - h_j‖² .
                        Baja = embeddings vecinos parecidos = smoothing.

Métricas de explicabilidad (heredadas):
  - compute_fidelity(): Fidelity+/- via GNNExplainer.
  - explanation_sparsity(): fracción de aristas no importantes.
  - permutation_test(): ¿el modelo usa la información del nodo fuente?
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
import numpy as np


# ===========================================================================
# OVER-SQUASHING: Jacobian norms
# ===========================================================================

def jacobian_norm_by_distance(model, data: Data, num_nodes: int = None,
                                graph_level: bool = False) -> dict:
    """
    Para cada par (i,j), calcula ‖∂h_i^L/∂x_j‖ y agrupa por dist(i,j).
    Para graph_level=True, usa los logits del grafo entero.
    """
    from rewiring import _shortest_paths
    num_nodes = num_nodes or data.num_nodes
    dist_matrix = _shortest_paths(data.edge_index, num_nodes)
    model.eval()
    by_dist = {}

    if graph_level:
        # Una sola pasada: ‖∂y_graph/∂x_j‖ por cada j
        x_in = data.x.clone().detach().requires_grad_(True)
        batch = torch.zeros(num_nodes, dtype=torch.long)
        logits = model(x_in, data.edge_index, batch)
        logits.sum().backward()
        grad = x_in.grad
        norms = grad.norm(dim=1).detach().cpu().numpy()
        # En graph-level no hay "nodo i" único — saltar
        return {-1: float(norms.mean())}

    for i in range(num_nodes):
        x_in = data.x.clone().detach().requires_grad_(True)
        logits = model(x_in, data.edge_index)
        logits[i].sum().backward()
        grad  = x_in.grad[:num_nodes]
        norms = grad.norm(dim=1).detach().cpu().numpy()

        for j in range(num_nodes):
            if i == j:
                continue
            d = dist_matrix[i, j]
            if np.isinf(d):
                continue
            by_dist.setdefault(int(d), []).append(norms[j])

    return {d: float(np.mean(v)) for d, v in sorted(by_dist.items())}


# ===========================================================================
# OVER-SMOOTHING: MAD y Dirichlet energy
# ===========================================================================

@torch.no_grad()
def mean_average_distance(embeddings: torch.Tensor) -> float:
    """
    MAD = media de las distancias coseno entre todos los pares de nodos.
    Rango: [0, 1]. Bajo = embeddings homogéneos = over-smoothing.

    Implementación O(N²) — fina para N < ~5000.
    """
    if embeddings.dim() != 2 or embeddings.shape[0] < 2:
        return 0.0
    emb = embeddings.detach().cpu()
    emb_norm = F.normalize(emb, p=2, dim=1)
    cos_sim = emb_norm @ emb_norm.t()                     # (N, N) en [-1, 1]
    cos_dist = 1.0 - cos_sim                              # (N, N) en [0, 2]
    N = emb.shape[0]
    # Excluye diagonal
    mask = ~torch.eye(N, dtype=torch.bool)
    return float(cos_dist[mask].mean().item())


@torch.no_grad()
def dirichlet_energy(embeddings: torch.Tensor, edge_index: torch.Tensor) -> float:
    """
    E_dir = (1/|E|) * Σ_{(i,j) ∈ E} ‖h_i - h_j‖²
    Baja = vecinos parecidos = over-smoothing.

    Promediada por arista para ser comparable entre grafos de tamaño distinto.
    """
    if edge_index.shape[1] == 0:
        return 0.0
    emb = embeddings.detach().cpu()
    src = edge_index[0].cpu()
    dst = edge_index[1].cpu()
    diff = emb[src] - emb[dst]                            # (E, D)
    sq   = diff.pow(2).sum(dim=1)                         # (E,)
    return float(sq.mean().item())


@torch.no_grad()
def get_node_embeddings(model, data: Data, graph_level: bool = False) -> torch.Tensor:
    """
    Extrae los embeddings de nodos de la PENÚLTIMA capa (antes del clasificador).
    Para modelos GAT a nivel de nodo: salida de la última GATConv.
    Para GATGraphLevel: salida de la última GATConv ANTES del global_pool.
    """
    model.eval()
    if graph_level:
        # Hace forward manual hasta antes del pooling
        x = data.x
        edge_index = data.edge_index
        for i, (conv, bn) in enumerate(zip(model.convs, model.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            if i < len(model.convs) - 1:
                x = F.elu(x)
        return x  # (N, hidden) — embeddings antes del pooling
    else:
        # Para GAT estándar: salida de la última GATConv
        x = data.x
        edge_index = data.edge_index
        for i, conv in enumerate(model.convs):
            x = conv(x, edge_index)
            if i < len(model.convs) - 1:
                x = F.elu(x)
        return x


# ===========================================================================
# EXPLICABILIDAD: GNNExplainer (heredado, simplificado)
# ===========================================================================

def _make_explainer(model, graph_level: bool = False):
    task_level = 'graph' if graph_level else 'node'
    return Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=100),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level=task_level,
            return_type='raw',
        ),
    )


def compute_fidelity(model, data: Data, target_nodes: list,
                      graph_level: bool = False) -> dict:
    model.eval()
    explainer = _make_explainer(model, graph_level)
    results   = {}

    with torch.no_grad():
        if graph_level:
            batch      = torch.zeros(data.num_nodes, dtype=torch.long)
            logits_f   = model(data.x, data.edge_index, batch)
        else:
            logits_f   = model(data.x, data.edge_index)
        probs_full = F.softmax(logits_f, dim=-1)

    for node_idx in target_nodes:
        try:
            if graph_level:
                explanation = explainer(data.x, data.edge_index)
            else:
                explanation = explainer(data.x, data.edge_index, index=node_idx)

            edge_mask = explanation.edge_mask
            important = edge_mask > 0.5
            ei_plus   = data.edge_index[:, important]
            ei_minus  = data.edge_index[:, ~important]

            if ei_plus.shape[1] == 0 or ei_minus.shape[1] == 0:
                results[node_idx] = {'fidelity_plus': 0.0, 'fidelity_minus': 0.0}
                continue

            with torch.no_grad():
                if graph_level:
                    batch    = torch.zeros(data.num_nodes, dtype=torch.long)
                    p_plus   = F.softmax(model(data.x, ei_plus,  batch), dim=-1)
                    p_minus  = F.softmax(model(data.x, ei_minus, batch), dim=-1)
                    pred_cls = probs_full[0].argmax().item()
                    p_f, p_p, p_m = (probs_full[0, pred_cls].item(),
                                     p_plus[0,  pred_cls].item(),
                                     p_minus[0, pred_cls].item())
                else:
                    p_plus   = F.softmax(model(data.x, ei_plus),  dim=-1)
                    p_minus  = F.softmax(model(data.x, ei_minus), dim=-1)
                    pred_cls = probs_full[node_idx].argmax().item()
                    p_f, p_p, p_m = (probs_full[node_idx, pred_cls].item(),
                                     p_plus[node_idx,  pred_cls].item(),
                                     p_minus[node_idx, pred_cls].item())

            results[node_idx] = {
                'fidelity_plus':  p_f - p_m,
                'fidelity_minus': p_f - p_p,
            }
        except Exception as e:
            print(f"  [warn] Explainer failed for node {node_idx}: {e}")
            results[node_idx] = {'fidelity_plus': 0.0, 'fidelity_minus': 0.0}

    return results


def permutation_test(model, data: Data, target_nodes: list,
                      distant_nodes: list, n_permutations: int = 20) -> dict:
    model.eval()
    with torch.no_grad():
        preds    = model(data.x, data.edge_index).argmax(dim=-1)
        base_acc = sum(preds[n].item() == data.y[n].item()
                       for n in target_nodes) / len(target_nodes)

    drops = []
    for _ in range(n_permutations):
        x_perm = data.x.clone()
        perm   = torch.randperm(len(distant_nodes)).tolist()
        x_perm[distant_nodes] = x_perm[[distant_nodes[p] for p in perm]]
        with torch.no_grad():
            preds_p  = model(x_perm, data.edge_index).argmax(dim=-1)
            perm_acc = sum(preds_p[n].item() == data.y[n].item()
                           for n in target_nodes) / len(target_nodes)
        drops.append(base_acc - perm_acc)

    return {'base_acc': base_acc,
            'mean_acc_drop': float(np.mean(drops)),
            'std_acc_drop':  float(np.std(drops))}


def explanation_sparsity(model, data: Data, target_nodes: list,
                          graph_level: bool = False) -> float:
    explainer  = _make_explainer(model, graph_level)
    sparsities = []
    for node_idx in target_nodes:
        try:
            if graph_level:
                explanation = explainer(data.x, data.edge_index)
            else:
                explanation = explainer(data.x, data.edge_index, index=node_idx)
            mask = explanation.edge_mask
            sparsities.append((mask < 0.5).float().mean().item())
        except Exception:
            pass
    return float(np.mean(sparsities)) if sparsities else 0.0
