from __future__ import annotations

from typing import Any

import torch
from torch_geometric.nn import HeteroConv, SAGEConv

from .base import BaseHeteroEncoder
from .input_projection import HeteroInputProjection


class HeteroSAGEMatchedEncoder(BaseHeteroEncoder):
    """HeteroSAGE with the same input projection / norms as attention models."""

    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        aggr: str = "sum",
        residual: bool = True,
    ) -> None:
        super().__init__()

        node_types, edge_types = metadata
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)
        self.residual = residual

        if not self.edge_types:
            raise ValueError("HeteroSAGEMatchedEncoder requires at least one edge type.")

        self.input_projection = HeteroInputProjection(
            node_types=self.node_types,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        for _ in range(num_layers):
            conv_dict = {
                edge_type: SAGEConv(
                    (hidden_dim, hidden_dim),
                    hidden_dim,
                    root_weight=False,
                )
                for edge_type in self.edge_types
            }
            self.convs.append(HeteroConv(conv_dict, aggr=aggr))
            self.norms.append(
                torch.nn.ModuleDict(
                    {
                        node_type: torch.nn.LayerNorm(hidden_dim)
                        for node_type in self.node_types
                    }
                )
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
                h = updated.get(node_type, previous[node_type])
                h = self.norms[layer_idx][node_type](h)
                h = torch.relu(h)
                h = self.dropout(h)
                if self.residual:
                    h = h + previous[node_type]
                next_h[node_type] = h

            h_dict = next_h

        return h_dict
