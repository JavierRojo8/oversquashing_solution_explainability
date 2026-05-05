"""
Training and evaluation para clasificación de nodos y grafos.

Modos:
  - Single-graph (Cora): un Data con máscaras globales.
  - Multi-graph (RingTransfer): lista de Data, cada uno con su máscara.
  - Graph-level (MNIST Superpíxeles): DataLoader con etiqueta por grafo.

Device: detecta automáticamente MPS (Mac GPU) > CUDA > CPU.
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Prioridad: CUDA → MPS (Apple Silicon) → CPU."""
    if torch.cuda.is_available():
        dev = torch.device('cuda')
        print(f"  [device] GPU detectada: {torch.cuda.get_device_name(0)}")
        return dev
    if torch.backends.mps.is_available():
        return torch.device('mps')
    print("  [device] CUDA no disponible — usando CPU. "
          "Instala PyTorch con soporte CUDA: https://pytorch.org/get-started/locally/")
    return torch.device('cpu')


# ---------------------------------------------------------------------------
# Single-graph training  (Cora)
# ---------------------------------------------------------------------------

def train_epoch(model, data: Data, optimizer) -> float:
    model.train()
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index)
    loss   = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data: Data, mask: torch.Tensor) -> float:
    model.eval()
    logits = model(data.x, data.edge_index)
    pred   = logits[mask].argmax(dim=-1)
    return (pred == data.y[mask]).sum().item() / mask.sum().item()


def train_single_graph(
    model,
    data: Data,
    epochs: int = 300,
    lr: float = 0.005,
    weight_decay: float = 5e-4,
    verbose: bool = True,
) -> dict:
    """Entrena sobre un único grafo con máscaras train/val/test."""
    device    = get_device()
    model     = model.to(device)
    data      = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val, best_test = 0.0, 0.0

    for epoch in range(1, epochs + 1):
        loss     = train_epoch(model, data, optimizer)
        val_acc  = evaluate(model, data, data.val_mask)
        test_acc = evaluate(model, data, data.test_mask)

        if val_acc > best_val:
            best_val  = val_acc
            best_test = test_acc

        if verbose and epoch % 50 == 0:
            print(f"  Epoch {epoch:03d} | Loss {loss:.4f} | "
                  f"Val {val_acc:.4f} | Test {test_acc:.4f}")

    return {'val_acc': best_val, 'test_acc': best_test, 'model': model}


# ---------------------------------------------------------------------------
# Multi-graph training  (RingTransfer)
# ---------------------------------------------------------------------------

def train_graph_list(
    model_cls,
    graphs: list,
    model_kwargs: dict,
    epochs: int = 200,
    lr: float = 0.005,
    weight_decay: float = 5e-4,
    verbose: bool = True,
) -> dict:
    """
    Entrena un modelo sobre una lista de grafos (estilo RingTransfer).
    num_nodes es constante — el rewiring nunca añade nodos.
    """
    device = get_device()

    g0          = graphs[0]
    num_classes = g0.num_classes if hasattr(g0, 'num_classes') else int(g0.y.max().item()) + 1
    in_channels = g0.x.shape[1]

    model     = model_cls(in_channels=in_channels, out_channels=num_classes,
                          **model_kwargs).to(device)
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
            loss   = F.cross_entropy(logits[g.train_mask], g.y[g.train_mask])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 20 == 0 or epoch == epochs:
            val_acc  = _acc_over_graphs(model, val_graphs,  'val_mask')
            test_acc = _acc_over_graphs(model, test_graphs, 'test_mask')
            if val_acc > best_val:
                best_val  = val_acc
                best_test = test_acc
            if verbose:
                print(f"  Epoch {epoch:03d} | Loss {total_loss/max(len(train_graphs),1):.4f} | "
                      f"Val {val_acc:.4f} | Test {test_acc:.4f}")

    return {'val_acc': best_val, 'test_acc': best_test, 'model': model}


@torch.no_grad()
def _acc_over_graphs(model, graphs: list, mask_attr: str) -> float:
    model.eval()
    correct, total = 0, 0
    for g in graphs:
        mask = getattr(g, mask_attr)
        if not mask.any():
            continue
        logits   = model(g.x, g.edge_index)
        pred     = logits[mask].argmax(dim=-1)
        correct += (pred == g.y[mask]).sum().item()
        total   += mask.sum().item()
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Graph-level training  (MNIST Superpíxeles)
# ---------------------------------------------------------------------------

def train_graph_level(
    model_cls,
    train_dataset,
    val_dataset,
    test_dataset,
    model_kwargs: dict,
    epochs: int = 30,
    lr: float = 0.001,
    weight_decay: float = 5e-4,
    batch_size: int = 64,
    verbose: bool = True,
) -> dict:
    """
    Entrena para clasificación a nivel de grafo completo (MNIST Superpíxeles).
    Usa DataLoader de PyG con batching automático.
    Requiere que el modelo tenga global pooling integrado.
    """
    device = get_device()
    print(f"  [device] Usando: {device}")

    g0          = train_dataset[0]
    in_channels = g0.x.shape[1]
    all_labels = [int(d.y.item()) for d in train_dataset]
    all_labels += [int(d.y.item()) for d in val_dataset]
    all_labels += [int(d.y.item()) for d in test_dataset]
    num_classes = max(all_labels) + 1

    model     = model_cls(in_channels=in_channels, out_channels=num_classes,
                          **model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # MPS no soporta num_workers > 0 de forma estable
    num_workers = 0

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    best_val, best_test = 0.0, 0.0

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        total_loss, total_graphs = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.batch)
            y = batch.y.view(-1).long()
            loss   = F.cross_entropy(logits, y)
            loss.backward()
            # Gradient clipping — útil con MPS
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss   += loss.item() * batch.num_graphs
            total_graphs += batch.num_graphs

        scheduler.step()

        # --- eval ---
        val_acc  = _acc_graph_level(model, val_loader,  device)
        test_acc = _acc_graph_level(model, test_loader, device)

        if val_acc > best_val:
            best_val  = val_acc
            best_test = test_acc

        if epoch == 1:
            print(batch.x.shape)
            print(batch.y.shape, batch.y[:10])
            print(logits.shape)

        if verbose:
            avg_loss = total_loss / max(total_graphs, 1)
            print(f"  Epoch {epoch:02d}/{epochs} | Loss {avg_loss:.4f} | "
                  f"Val {val_acc:.4f} | Test {test_acc:.4f}")

    return {'val_acc': best_val, 'test_acc': best_test, 'model': model}


@torch.no_grad()
def _acc_graph_level(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        batch    = batch.to(device)
        logits   = model(batch.x, batch.edge_index, batch.batch)
        pred     = logits.argmax(dim=-1)
        y = batch.y.view(-1).long()
        correct += (pred == y).sum().item()
        total   += batch.num_graphs
    return correct / total if total > 0 else 0.0
