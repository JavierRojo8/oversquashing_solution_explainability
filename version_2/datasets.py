"""
Datasets para experimentos LRE-GAT:
  - RingTransfer: dataset sintético para over-squashing
  - MNISTSuperpixels: clasificación de dígitos como grafos de superpíxeles
"""

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, MNISTSuperpixels
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T


# ---------------------------------------------------------------------------
# RingTransfer
# ---------------------------------------------------------------------------

def make_ring_transfer(num_nodes: int = 20, num_classes: int = 5,
                        num_graphs: int = 500, noise_std: float = 0.1,
                        seed: int = 42) -> list[Data]:
    """
    Anillo de num_nodes nodos. Nodo 0 tiene señal (one-hot clase + ruido).
    Target: nodo L//2 (a 10 hops). Diseñado para exponer over-squashing.
    """
    torch.manual_seed(seed)
    graphs = []

    src        = torch.arange(num_nodes, dtype=torch.long)
    dst        = torch.roll(src, -1)
    edge_index = to_undirected(torch.stack([src, dst], dim=0))

    target  = num_nodes // 2
    n_train = int(0.70 * num_graphs)
    n_val   = int(0.85 * num_graphs)

    for idx in range(num_graphs):
        cls = torch.randint(0, num_classes, (1,)).item()
        x   = torch.randn(num_nodes, num_classes) * noise_std
        x[0] = 0.0
        x[0, cls] = 1.0
        y = torch.full((num_nodes,), cls, dtype=torch.long)

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask   = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask  = torch.zeros(num_nodes, dtype=torch.bool)

        if idx < n_train:
            train_mask[target] = True
        elif idx < n_val:
            val_mask[target] = True
        else:
            test_mask[target] = True

        graphs.append(Data(
            x=x, edge_index=edge_index.clone(), y=y,
            train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
            num_classes=num_classes,
        ))

    return graphs


# ---------------------------------------------------------------------------
# MNIST Superpíxeles
# ---------------------------------------------------------------------------

class NormalizeMNIST(T.BaseTransform):
    """
    Normaliza features a rangos coherentes:
      - Intensidad: ya viene en [0, 1] desde MNISTSuperpixels.
      - Posición:   se centra y reescala a [-1, 1] (no [0, 1]) para que el
                    centro de la imagen sea 0. Esto evita que el mean pooling
                    arrastre un sesgo geométrico hacia (0.5, 0.5) y hace que
                    las coordenadas tengan media cero, mejorando la
                    convergencia de GAT.
    """
    def forward(self, data: Data) -> Data:
        data.x = data.x.float()

        if hasattr(data, "pos") and data.pos is not None:
            pos = data.pos.float()
            # MNISTSuperpixels usa pos en [0, 27]
            if pos.max() > 1.5:
                pos = (pos / 27.0) * 2.0 - 1.0   # → [-1, 1]
            # x = [intensity, pos_x, pos_y]  shape (N, 3)
            data.x = torch.cat([data.x, pos], dim=1)

        # MNIST: y es escalar por grafo
        data.y = data.y.long().view(-1)
        return data

def load_mnist_superpixels(
    root: str = "./data",
    subset_train: int = 10000,
    subset_test: int = 2000,
) -> tuple:
    """
    Carga MNIST Superpíxeles (~75 nodos por grafo, 10 clases).

    Args:
        root:         Directorio donde descargar los datos.
        subset_train: Número de grafos de entrenamiento (max 60000).
        subset_test:  Número de grafos de test (max 10000).

    Returns:
        (train_dataset, val_dataset, test_dataset)
        val = 20% del subconjunto de train
    """
    transform = NormalizeMNIST()

    train_full = MNISTSuperpixels(root=root, train=True,  transform=transform)
    test_full  = MNISTSuperpixels(root=root, train=False, transform=transform)

    # Subset para velocidad en CPU/MPS
    subset_train = min(subset_train, len(train_full))
    subset_test  = min(subset_test,  len(test_full))

    indices_train = torch.randperm(len(train_full))[:subset_train]
    train_sub     = train_full.index_select(indices_train)

    # Split train → 80% train, 20% val
    n_val   = int(0.2 * subset_train)
    n_train = subset_train - n_val
    train_dataset = train_sub[:n_train]
    val_dataset   = train_sub[n_train:]

    test_dataset  = test_full.index_select(
        torch.randperm(len(test_full))[:subset_test]
    )

    print(f"  [MNIST] Train: {len(train_dataset)} | "
          f"Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    print(f"  [MNIST] Nodos por grafo: ~{train_dataset[0].num_nodes} | "
          f"Features: {train_dataset[0].x.shape[1]} | Clases: 10")

    return train_dataset, val_dataset, test_dataset


def load_cora(root: str = "./data") -> Data:
    """Carga Cora con split estándar y normalización de features."""
    dataset = Planetoid(root=root, name="Cora", transform=T.NormalizeFeatures())
    return dataset[0]
