from __future__ import annotations

from typing import Any

import torch
from torch_geometric.nn import GATv2Conv, HeteroConv

from .base import BaseHeteroEncoder
from .input_projection import HeteroInputProjection


class HeteroGATv2Encoder(BaseHeteroEncoder):
    """Per-relation GATv2 inside HeteroConv (+ shared input projection)."""

    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        hidden_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.2,
        aggr: str = "sum",
        residual: bool = True,
    ) -> None:
        super().__init__()

        node_types, edge_types = metadata
        if hidden_dim % heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by heads when concat=True."
            )

        self.node_types = list(node_types)
        self.edge_types = list(edge_types)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.residual = residual

        if not self.edge_types:
            raise ValueError("HeteroGATv2Encoder requires at least one edge type.")

        self.input_projection = HeteroInputProjection(
            node_types=self.node_types,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        out_per_head = hidden_dim // heads
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        for _ in range(num_layers):
            conv_dict = {
                edge_type: GATv2Conv(
                    in_channels=(hidden_dim, hidden_dim),
                    out_channels=out_per_head,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    add_self_loops=False,
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
                h = updated[node_type] if node_type in updated else previous[node_type]
                h = self.norms[layer_idx][node_type](h)
                h = torch.relu(h)
                h = self.dropout(h)
                if self.residual:
                    h = h + previous[node_type]
                next_h[node_type] = h

            h_dict = next_h

        return h_dict
