"""
Training and evaluation for node classification.
Handles both single-graph (Cora) and multi-graph (RingTransfer) settings.
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data


def train_epoch(model, data: Data, optimizer) -> float:
    model.train()
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index)
    mask = data.train_mask
    loss = F.cross_entropy(logits[mask], data.y[mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data: Data, mask: torch.Tensor) -> float:
    model.eval()
    logits = model(data.x, data.edge_index)
    pred = logits[mask].argmax(dim=-1)
    correct = (pred == data.y[mask]).sum().item()
    return correct / mask.sum().item()


def train_single_graph(model, data: Data, epochs: int = 300,
                        lr: float = 0.005, weight_decay: float = 5e-4,
                        verbose: bool = True) -> dict:
    """Train on a single graph (Cora-style)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val, best_test = 0.0, 0.0

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, data, optimizer)
        val_acc = evaluate(model, data, data.val_mask)
        test_acc = evaluate(model, data, data.test_mask)

        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc

        if verbose and epoch % 50 == 0:
            print(f"  Epoch {epoch:03d} | Loss {loss:.4f} | Val {val_acc:.4f} | Test {test_acc:.4f}")

    return {'val_acc': best_val, 'test_acc': best_test, 'model': model}


def train_graph_list(model_cls, graphs: list, model_kwargs: dict,
                      epochs: int = 200, lr: float = 0.005,
                      weight_decay: float = 5e-4, verbose: bool = True) -> dict:
    """
    Train on a list of graphs (RingTransfer-style).
    One model per experiment, batching graphs via manual loop.
    Returns aggregate accuracy over test graphs.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Infer model dimensions from first graph
    g0 = graphs[0]
    num_classes = g0.num_classes if hasattr(g0, 'num_classes') else int(g0.y.max().item()) + 1
    in_channels = g0.x.shape[1]

    model = model_cls(in_channels=in_channels, out_channels=num_classes, **model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_graphs = [g.to(device) for g in graphs if g.train_mask.any()]
    val_graphs   = [g.to(device) for g in graphs if g.val_mask.any()]
    test_graphs  = [g.to(device) for g in graphs if g.test_mask.any()]

    best_val, best_test = 0.0, 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for g in train_graphs:
            optimizer.zero_grad()
            logits = model(g.x, g.edge_index)
            loss = F.cross_entropy(logits[g.train_mask], g.y[g.train_mask])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 20 == 0 or epoch == epochs:
            val_acc = _acc_over_graphs(model, val_graphs, 'val_mask')
            test_acc = _acc_over_graphs(model, test_graphs, 'test_mask')
            if val_acc > best_val:
                best_val = val_acc
                best_test = test_acc
            if verbose:
                print(f"  Epoch {epoch:03d} | Loss {total_loss/len(train_graphs):.4f} "
                      f"| Val {val_acc:.4f} | Test {test_acc:.4f}")

    return {'val_acc': best_val, 'test_acc': best_test, 'model': model}


@torch.no_grad()
def _acc_over_graphs(model, graphs, mask_attr):
    model.eval()
    correct, total = 0, 0
    for g in graphs:
        mask = getattr(g, mask_attr)
        if not mask.any():
            continue
        logits = model(g.x, g.edge_index)
        pred = logits[mask].argmax(dim=-1)
        correct += (pred == g.y[mask]).sum().item()
        total += mask.sum().item()
    return correct / total if total > 0 else 0.0
