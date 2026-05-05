"""
Diagnostic metrics: Jacobian Norm, Fidelity+/-, Permutation Test.
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
import numpy as np


# ---------------------------------------------------------------------------
# Jacobian Norm
# ---------------------------------------------------------------------------

def compute_jacobian_norm(model, data: Data, target_nodes: list[int],
                           source_nodes: list[int]) -> dict:
    """
    Compute ||dh_i^L / dx_j|| for each (i, j) pair.
    Returns dict mapping (i, j) -> scalar norm.
    """
    model.eval()
    results = {}

    for i in target_nodes:
        # Need gradient w.r.t. x
        x = data.x.clone().detach().requires_grad_(True)
        logits = model(x, data.edge_index)
        # Use sum of logits at node i as scalar output
        output = logits[i].sum()
        output.backward()
        grad = x.grad  # (N, F)

        for j in source_nodes:
            norm = grad[j].norm().item()
            results[(i, j)] = norm

        # Reset grad
        if x.grad is not None:
            x.grad.zero_()

    return results


def jacobian_norm_by_distance(model, data: Data, num_nodes: int = None) -> dict:
    """
    Compute average Jacobian norm grouped by hop distance.
    Returns {distance: mean_norm}.
    """
    from oversquashing_solution_explainability.version_2.rewiring import _pairwise_distances
    if num_nodes is None:
        num_nodes = data.num_nodes

    # Skip virtual node if present
    real_n = getattr(data, 'original_num_nodes', num_nodes)
    dist_matrix = _pairwise_distances(data.edge_index, real_n)

    model.eval()

    # Full Jacobian for real nodes only: forward once per target node
    by_dist = {}
    for i in range(real_n):
        x_in = data.x.clone().detach().requires_grad_(True)
        logits = model(x_in, data.edge_index)
        logits[i].sum().backward()
        grad = x_in.grad[:real_n]  # (real_n, F)
        norms = grad.norm(dim=1).detach().cpu().numpy()  # (real_n,)

        for j in range(real_n):
            if i == j:
                continue
            d = dist_matrix[i, j]
            if np.isinf(d):
                continue
            d = int(d)
            if d not in by_dist:
                by_dist[d] = []
            by_dist[d].append(norms[j])

    return {d: float(np.mean(v)) for d, v in sorted(by_dist.items())}


# ---------------------------------------------------------------------------
# GNNExplainer-based Fidelity
# ---------------------------------------------------------------------------

def compute_fidelity(model, data: Data, target_nodes: list[int]) -> dict:
    """
    Returns Fidelity+ and Fidelity- for each target node using GNNExplainer.
    Fidelity+ = drop in probability when explanation subgraph is KEPT (complement removed).
    Fidelity- = drop when explanation subgraph is REMOVED.
    """
    model.eval()
    explainer = Explainer(
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

    results = {}
    with torch.no_grad():
        logits_full = model(data.x, data.edge_index)
        probs_full = F.softmax(logits_full, dim=-1)

    for node_idx in target_nodes:
        try:
            explanation = explainer(data.x, data.edge_index, index=node_idx)
            edge_mask = explanation.edge_mask  # (E,)

            # Fidelity+: remove unimportant edges (keep explanation)
            important = edge_mask > 0.5
            ei_plus = data.edge_index[:, important]
            if ei_plus.shape[1] == 0:
                results[node_idx] = {'fidelity_plus': 0.0, 'fidelity_minus': 0.0}
                continue
            with torch.no_grad():
                logits_plus = model(data.x, ei_plus)
                probs_plus = F.softmax(logits_plus, dim=-1)

            # Fidelity-: remove important edges (keep unimportant)
            unimportant = ~important
            ei_minus = data.edge_index[:, unimportant]
            if ei_minus.shape[1] == 0:
                results[node_idx] = {'fidelity_plus': 0.0, 'fidelity_minus': 0.0}
                continue
            with torch.no_grad():
                logits_minus = model(data.x, ei_minus)
                probs_minus = F.softmax(logits_minus, dim=-1)

            pred_class = probs_full[node_idx].argmax().item()
            p_full = probs_full[node_idx, pred_class].item()
            p_plus = probs_plus[node_idx, pred_class].item()
            p_minus = probs_minus[node_idx, pred_class].item()

            results[node_idx] = {
                'fidelity_plus': p_full - p_minus,   # drop when explanation removed
                'fidelity_minus': p_full - p_plus,   # drop when non-explanation removed
            }
        except Exception as e:
            print(f"  [warn] Explainer failed for node {node_idx}: {e}")
            results[node_idx] = {'fidelity_plus': 0.0, 'fidelity_minus': 0.0}

    return results


# ---------------------------------------------------------------------------
# Permutation Test
# ---------------------------------------------------------------------------

def permutation_test(model, data: Data, target_nodes: list[int],
                      distant_nodes: list[int], n_permutations: int = 20) -> dict:
    """
    Shuffle features of distant_nodes and measure accuracy change on target_nodes.
    Large drop → model uses distant info. Small drop → info not used (over-squashing).
    Returns {'mean_acc_drop': float, 'std_acc_drop': float}.
    """
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        preds = logits.argmax(dim=-1)
        base_correct = sum(preds[n].item() == data.y[n].item() for n in target_nodes)
        base_acc = base_correct / len(target_nodes)

    drops = []
    for _ in range(n_permutations):
        x_perm = data.x.clone()
        perm_idx = torch.randperm(len(distant_nodes))
        perm_nodes = [distant_nodes[i] for i in perm_idx]
        x_perm[distant_nodes] = x_perm[perm_nodes]

        with torch.no_grad():
            logits_p = model(x_perm, data.edge_index)
            preds_p = logits_p.argmax(dim=-1)
            perm_correct = sum(preds_p[n].item() == data.y[n].item() for n in target_nodes)
            perm_acc = perm_correct / len(target_nodes)

        drops.append(base_acc - perm_acc)

    return {
        'base_acc': base_acc,
        'mean_acc_drop': float(np.mean(drops)),
        'std_acc_drop': float(np.std(drops)),
    }


# ---------------------------------------------------------------------------
# Explanation Sparsity
# ---------------------------------------------------------------------------

def explanation_sparsity(model, data: Data, target_nodes: list[int]) -> float:
    """Fraction of edges masked out by GNNExplainer (closer to 1 = sparser)."""
    explainer = Explainer(
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
    sparsities = []
    for node_idx in target_nodes:
        try:
            explanation = explainer(data.x, data.edge_index, index=node_idx)
            mask = explanation.edge_mask
            sparsities.append((mask < 0.5).float().mean().item())
        except Exception:
            pass
    return float(np.mean(sparsities)) if sparsities else 0.0
