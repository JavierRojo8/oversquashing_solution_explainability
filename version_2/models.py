"""
GAT model — multi-head attention, configurable depth.

Dos variantes:
  - GAT:          clasificación a nivel de nodo (RingTransfer, Cora)
  - GATGraphLevel: clasificación a nivel de grafo (MNIST Superpíxeles)
                   añade global pooling (mean + max) sobre los embeddings de nodos.

Extracción de attention weights via return_attention_weights=True (booleano),
compatible con todas las versiones de PyG >= 2.0.

NOTA SOBRE DROPOUT:
GATConv tiene dos sitios donde el dropout actúa:
  1. El parámetro `dropout=` de GATConv aplica dropout a los attention weights αᵢⱼ.
  2. F.dropout(x, ...) en el forward aplica dropout a las features.
Aplicar ambos a la vez con valores típicos (0.5 + 0.6) destruye el aprendizaje en
grafos pequeños como MNIST-Superpíxeles. Aquí dejamos GATConv SIN dropout interno
y controlamos solo F.dropout en el forward.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from typing import Union


# ---------------------------------------------------------------------------
# GAT — clasificación de nodos
# ---------------------------------------------------------------------------

class GAT(nn.Module):
    def __init__(self, in_channels: int, hidden: int, out_channels: int,
                 heads: int = 4, num_layers: int = 3, dropout: float = 0.6):
        super().__init__()
        assert num_layers >= 2
        self.dropout    = dropout
        self.num_layers = num_layers
        self.convs      = nn.ModuleList()

        self.convs.append(
            GATConv(in_channels, hidden, heads=heads,
                    dropout=dropout, add_self_loops=True)
        )
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden * heads, hidden, heads=heads,
                        dropout=dropout, add_self_loops=True)
            )
        self.convs.append(
            GATConv(hidden * heads, out_channels, heads=1, concat=False,
                    dropout=dropout, add_self_loops=True)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> Union[torch.Tensor, tuple]:
        if return_attention_weights:
            return self._forward_with_attention(x, edge_index)
        return self._forward_normal(x, edge_index)

    def _forward_normal(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            if self.training:
                x = F.dropout(x, p=self.dropout, training=True)
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
        return x

    def _forward_with_attention(self, x, edge_index):
        attention_list = []
        for i, conv in enumerate(self.convs):
            if self.training:
                x = F.dropout(x, p=self.dropout, training=True)
            result   = conv(x, edge_index, return_attention_weights=True)
            x        = result[0]
            ei_layer = result[1][0]   # (2, E)
            alpha    = result[1][1]   # (E, H)
            attention_list.append((ei_layer, alpha.detach()))
            if i < len(self.convs) - 1:
                x = F.elu(x)
        return x, attention_list


# ---------------------------------------------------------------------------
# GATGraphLevel — clasificación de grafos (MNIST Superpíxeles)
# ---------------------------------------------------------------------------

class GATGraphLevel(nn.Module):
    """
    GAT con global pooling para clasificación a nivel de grafo (MNIST Superpíxeles).

    Diferencias clave respecto a la versión anterior:
      - GATConv SIN dropout interno (el dropout sobre attention rompe grafos pequeños).
      - BatchNorm1d entre capas para estabilizar el entrenamiento.
      - Pooling = concat(mean_pool, max_pool) → captura distribución y picos.
      - El dropout solo se aplica en el MLP final, no en cada GATConv.
      - num_layers por defecto = 4 (más expresividad para 10 clases).

    Arquitectura:
        GATConv → BN → ELU → ... → GATConv → mean_pool ⊕ max_pool → MLP → logits
    """

    def __init__(self, in_channels: int, hidden: int, out_channels: int,
                 heads: int = 4, num_layers: int = 4, dropout: float = 0.3):
        super().__init__()
        assert num_layers >= 2
        self.dropout = dropout
        self.convs   = nn.ModuleList()
        self.bns     = nn.ModuleList()

        # Primera capa
        self.convs.append(
            GATConv(in_channels, hidden, heads=heads,
                    dropout=0.0, add_self_loops=True)
        )
        self.bns.append(nn.BatchNorm1d(hidden * heads))

        # Capas intermedias
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden * heads, hidden, heads=heads,
                        dropout=0.0, add_self_loops=True)
            )
            self.bns.append(nn.BatchNorm1d(hidden * heads))

        # Última capa GAT (concat=False → salida limpia para pooling)
        self.convs.append(
            GATConv(hidden * heads, hidden, heads=1, concat=False,
                    dropout=0.0, add_self_loops=True)
        )
        self.bns.append(nn.BatchNorm1d(hidden))

        # MLP de clasificación tras pooling: mean ⊕ max → 2*hidden de entrada
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_channels),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor = None,
        return_attention_weights: bool = False,
    ) -> Union[torch.Tensor, tuple]:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        attention_list = []

        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            if return_attention_weights:
                result   = conv(x, edge_index, return_attention_weights=True)
                x        = result[0]
                ei_layer = result[1][0]
                alpha    = result[1][1]
                attention_list.append((ei_layer, alpha.detach()))
            else:
                x = conv(x, edge_index)

            x = bn(x)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Pooling combinado: media + máximo
        x_mean = global_mean_pool(x, batch)
        x_max  = global_max_pool(x, batch)
        x      = torch.cat([x_mean, x_max], dim=-1)

        logits = self.classifier(x)

        if return_attention_weights:
            return logits, attention_list
        return logits


# ---------------------------------------------------------------------------
# Utilidad: attention por arista → matriz densa (N, N)
# ---------------------------------------------------------------------------

def attention_to_matrix(
    edge_index: torch.Tensor,
    alpha: torch.Tensor,
    num_nodes: int,
    aggr: str = "mean",
) -> torch.Tensor:
    alpha = alpha.detach().cpu()
    if alpha.dim() == 2:
        alpha_agg = alpha.mean(dim=1) if aggr == "mean" else alpha.max(dim=1).values
    else:
        alpha_agg = alpha

    A = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    A[edge_index[0].cpu(), edge_index[1].cpu()] = alpha_agg.float()
    return A
