from __future__ import annotations

import torch
from torch_geometric.nn import Linear


class HeteroInputProjection(torch.nn.Module):
    """Per-node-type projection into a shared hidden dimension."""

    def __init__(
        self,
        node_types: list[str],
        hidden_dim: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.projections = torch.nn.ModuleDict(
            {
                node_type: Linear(-1, hidden_dim)
                for node_type in node_types
            }
        )
        self.norms = torch.nn.ModuleDict(
            {
                node_type: torch.nn.LayerNorm(hidden_dim)
                for node_type in node_types
            }
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        output: dict[str, torch.Tensor] = {}
        for node_type, x in x_dict.items():
            h = self.projections[node_type](x)
            h = self.norms[node_type](h)
            h = torch.relu(h)
            h = self.dropout(h)
            output[node_type] = h
        return output
