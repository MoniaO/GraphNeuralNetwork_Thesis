"""Encoder HGT — jedyny backbone FINAL 14.08.

Co robi
-------
Wspólna projekcja wejść typów węzłów + warstwy HGTConv (residual, LayerNorm).
Zwraca słownik embeddingów per typ; LinkPredictor spłaszcza je do dekodera.

Co wolno zmieniać (nowy eksperyment)
------------------------------------
hidden_dim, num_layers, heads, dropout, activation — przez Hydra `model.hgt.*`.
hidden_dim musi być podzielne przez heads.

Czego nie ruszać dla FINAL 14.08
--------------------------------
h32, L2, heads=4, dropout=0.25, activation=leaky_relu, residual=True.
"""

from __future__ import annotations

from typing import Any

import torch
from torch_geometric.nn import HGTConv

from taskA.models.encoder.activations import build_activation

from .base import BaseHeteroEncoder
from .input_projection import HeteroInputProjection


class HGTEncoder(BaseHeteroEncoder):
    """Heterogeneous Graph Transformer encoder with shared input projection."""

    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        hidden_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.2,
        residual: bool = True,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        node_types, edge_types = metadata
        if hidden_dim % heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by heads: "
                f"{hidden_dim=} {heads=}"
            )
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        self.node_types = list(node_types)
        self.metadata = (list(node_types), list(edge_types))
        self.residual = residual
        self.activation_name = str(activation)

        if not edge_types:
            raise ValueError("HGTEncoder requires at least one edge type.")

        self.input_projection = HeteroInputProjection(
            node_types=self.node_types,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.convs = torch.nn.ModuleList(
            [
                HGTConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    metadata=self.metadata,
                    heads=heads,
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        node_type: torch.nn.LayerNorm(hidden_dim)
                        for node_type in self.node_types
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.activations = torch.nn.ModuleList(
            [build_activation(activation) for _ in range(num_layers)]
        )
        self.dropout = torch.nn.Dropout(dropout)

    def encode(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[Any, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        h_dict = self.input_projection(x_dict)

        for layer_idx, conv in enumerate(self.convs):
            previous = h_dict
            updated = conv(h_dict, edge_index_dict)
            next_h: dict[str, torch.Tensor] = {}

            for node_type in self.node_types:
                candidate = updated.get(node_type) if isinstance(updated, dict) else None
                if candidate is not None:
                    message_output = candidate
                else:
                    message_output = previous[node_type]

                # Activation → dropout → residual → LayerNorm (audit contract).
                message_output = self.activations[layer_idx](message_output)
                message_output = self.dropout(message_output)
                if (
                    self.residual
                    and message_output.shape == previous[node_type].shape
                ):
                    message_output = message_output + previous[node_type]
                next_h[node_type] = self.norms[layer_idx][node_type](message_output)

            h_dict = next_h

        return h_dict
