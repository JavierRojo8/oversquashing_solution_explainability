"""
Métricas de diagnóstico para over-squashing:
  - Jacobian Norm (por distancia y puntual)
  - Fidelity+/- via GNNExplainer
  - Permutation Test
  - Explanation Sparsity

El flag graph_level=True adapta GNNExplainer para modelos GATGraphLevel
(clasificación a nivel de grafo completo, como MNIST Superpíxeles).
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
import numpy as np


# ---------------------------------------------------------------------------
# Jacobian Norm
# ---------------------------------------------------------------------------

def compute_jacobian_norm(model, data: Data,
                           target_nodes: list, source_nodes: list) -> dict:
    model.eval()
    results = {}
    for i in target_nodes:
        x = data.x.clone().detach().requires_grad_(True)
        logits = model(x, data.edge_index)
        logits[i].sum().backward()
        for j in source_nodes:
            results[(i, j)] = x.grad[j].norm().item() if x.grad is not None else 0.0
        if x.grad is not None:
            x.grad.zero_()
    return results


def jacobian_norm_by_distance(model, data: Data, num_nodes: int = None) -> dict:
    from rewiring import _pairwise_distances
    num_nodes   = num_nodes or data.num_nodes
    dist_matrix = _pairwise_distances(data.edge_index, num_nodes)
    model.eval()
    by_dist = {}

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

def graph_level_jacobian_norm_by_distance(model, data: Data,
                                          reference_node: int = None) -> dict:
    """
    Jacobian norm para clasificación a nivel de grafo.

    Mide cuánto afecta cada nodo al logit predicho del grafo completo.
    Luego agrupa esas sensibilidades por distancia respecto a un nodo
    de referencia, normalmente el nodo central del grafo.
    """
    from rewiring import _pairwise_distances

    model.eval()

    num_nodes = data.num_nodes
    reference_node = reference_node if reference_node is not None else num_nodes // 2

    batch = torch.zeros(num_nodes, dtype=torch.long, device=data.x.device)

    x_in = data.x.clone().detach().requires_grad_(True)

    logits = model(x_in, data.edge_index, batch)
    pred_class = logits.argmax(dim=-1).item()

    logits[0, pred_class].backward()

    grad = x_in.grad
    if grad is None:
        return {}

    node_grad_norms = grad.norm(dim=1).detach().cpu().numpy()

    dist_matrix = _pairwise_distances(data.edge_index.cpu(), num_nodes)

    by_dist = {}
    for j in range(num_nodes):
        if j == reference_node:
            continue

        d = dist_matrix[reference_node, j]
        if np.isinf(d):
            continue

        by_dist.setdefault(int(d), []).append(node_grad_norms[j])

    return {
        d: float(np.mean(vals))
        for d, vals in sorted(by_dist.items())
    }


# ---------------------------------------------------------------------------
# GNNExplainer helpers
# ---------------------------------------------------------------------------

def _make_explainer(model, graph_level: bool = False):
    """
    Construye un Explainer adaptado al nivel de tarea.
    graph_level=True para modelos con global pooling (MNIST).
    """
    if graph_level:
        return Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=100),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='graph',
                return_type='raw',
            ),
        )
    return Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=100),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='raw',
        ),
    )


# ---------------------------------------------------------------------------
# Fidelity
# ---------------------------------------------------------------------------

def compute_fidelity(model, data: Data, target_nodes: list,
                      graph_level: bool = False) -> dict:
    model.eval()
    explainer = _make_explainer(model, graph_level)
    results   = {}

    # Para graph_level, probs sobre el grafo completo
    with torch.no_grad():
        if graph_level:
            batch      = torch.zeros(data.num_nodes, dtype=torch.long)
            logits_f   = model(data.x, data.edge_index, batch)
            probs_full = F.softmax(logits_f, dim=-1)
        else:
            logits_f   = model(data.x, data.edge_index)
            probs_full = F.softmax(logits_f, dim=-1)

    for node_idx in target_nodes:
        try:
            if graph_level:
                explanation = explainer(data.x, data.edge_index)
            else:
                explanation = explainer(data.x, data.edge_index, index=node_idx)

            edge_mask   = explanation.edge_mask
            important   = edge_mask > 0.5
            ei_plus     = data.edge_index[:, important]
            ei_minus    = data.edge_index[:, ~important]

            if ei_plus.shape[1] == 0 or ei_minus.shape[1] == 0:
                results[node_idx] = {'fidelity_plus': 0.0, 'fidelity_minus': 0.0}
                continue

            with torch.no_grad():
                if graph_level:
                    batch    = torch.zeros(data.num_nodes, dtype=torch.long)
                    p_plus   = F.softmax(model(data.x, ei_plus,  batch), dim=-1)
                    p_minus  = F.softmax(model(data.x, ei_minus, batch), dim=-1)
                    pred_cls = probs_full[0].argmax().item()
                    p_f  = probs_full[0, pred_cls].item()
                    p_p  = p_plus[0,  pred_cls].item()
                    p_m  = p_minus[0, pred_cls].item()
                else:
                    p_plus   = F.softmax(model(data.x, ei_plus),  dim=-1)
                    p_minus  = F.softmax(model(data.x, ei_minus), dim=-1)
                    pred_cls = probs_full[node_idx].argmax().item()
                    p_f  = probs_full[node_idx, pred_cls].item()
                    p_p  = p_plus[node_idx,  pred_cls].item()
                    p_m  = p_minus[node_idx, pred_cls].item()

            results[node_idx] = {
                'fidelity_plus':  p_f - p_m,
                'fidelity_minus': p_f - p_p,
            }
        except Exception as e:
            print(f"  [warn] Explainer failed for node {node_idx}: {e}")
            results[node_idx] = {'fidelity_plus': 0.0, 'fidelity_minus': 0.0}

    return results


# ---------------------------------------------------------------------------
# Permutation Test
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Explanation Sparsity
# ---------------------------------------------------------------------------

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
