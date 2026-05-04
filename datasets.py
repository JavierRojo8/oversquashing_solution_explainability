"""
RingTransfer synthetic dataset and Cora loader.

RingTransfer: L-node ring where node 0 holds a class label in its features.
The task is to predict the class of the antipodal node (node L//2).
Requires L//2 hops of information propagation — designed to expose over-squashing.
"""

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T


def make_ring_transfer(num_nodes: int = 20, num_classes: int = 5,
                        num_graphs: int = 500, noise_std: float = 0.1,
                        seed: int = 42) -> list[Data]:
    """
    Each graph: ring of `num_nodes` nodes.
    Node 0 feature = one-hot class label + noise.
    All other node features = pure noise.
    Label of every node = class at node 0 (so antipodal node L//2 has correct label).
    Only node L//2 is in the test mask; node 0 is the signal source.
    """
    torch.manual_seed(seed)
    graphs = []

    # Ring edges: 0→1, 1→2, ..., (L-1)→0  (then made undirected)
    src = torch.arange(num_nodes, dtype=torch.long)
    dst = torch.roll(src, -1)
    edge_index = to_undirected(torch.stack([src, dst], dim=0))

    target = num_nodes // 2
    n_train = int(0.7 * num_graphs)
    n_val   = int(0.85 * num_graphs)

    for idx in range(num_graphs):
        cls = torch.randint(0, num_classes, (1,)).item()
        x = torch.randn(num_nodes, num_classes) * noise_std
        x[0] = 0.0
        x[0, cls] = 1.0  # signal at node 0
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
            x=x,
            edge_index=edge_index.clone(),
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            num_classes=num_classes,
        ))

    return graphs


def load_cora(root: str = "./data") -> Data:
    """Load Cora with standard train/val/test split."""
    dataset = Planetoid(root=root, name="Cora", transform=T.NormalizeFeatures())
    return dataset[0]
