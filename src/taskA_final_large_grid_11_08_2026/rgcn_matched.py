"""Relation-aware R-GCN with per-node-type input projection (11.08.2026).

Avoids losing node-type identity by projecting each type before the shared
RGCN stack. Relations are never collapsed.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv

from models.TaskA.encoders.base import BaseHeteroEncoder
from models.TaskA.encoders.input_projection import HeteroInputProjection
from utils.task_a_layout import global_layout


class HeteroRGCNMatchedEncoder(BaseHeteroEncoder):
    """HeteroInputProjection → flatten → RGCNConv stack → per-type slices."""

    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        residual: bool = True,
        num_bases: int | str = 8,
    ) -> None:
        super().__init__()
        node_types, edge_types = metadata
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.residual = bool(residual)
        self.dropout = nn.Dropout(float(dropout))

        if not self.edge_types:
            raise ValueError("HeteroRGCNMatchedEncoder requires ≥1 edge type.")

        n_rel = len(self.edge_types)
        if isinstance(num_bases, str):
            if str(num_bases).lower() != "full":
                raise ValueError(f"Unknown num_bases={num_bases!r}")
            bases = n_rel
        else:
            bases = int(num_bases)
            if bases < 1:
                raise ValueError("num_bases must be ≥1 or 'full'")
            bases = min(bases, n_rel)
        self.num_bases = bases

        self.input_projection = HeteroInputProjection(
            node_types=self.node_types,
            hidden_dim=self.hidden_dim,
            dropout=float(dropout),
        )
        self.convs = nn.ModuleList(
            [
                RGCNConv(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    num_relations=n_rel,
                    num_bases=bases,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)]
        )
        self._offsets: dict[str, int] | None = None
        self._sizes: dict[str, int] | None = None

    def _type_sizes(self, data: Any) -> dict[str, int]:
        return {
            nt: int(data[nt].num_nodes)
            for nt in sorted(data.node_types)
        }

    def _flatten(
        self, data: Any, h_dict: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # global_layout → (offsets, name_lookup); sizes come from num_nodes.
        offsets, _lookup = global_layout(data)
        sizes = self._type_sizes(data)
        self._offsets, self._sizes = offsets, sizes
        parts = [h_dict[nt] for nt in sorted(data.node_types)]
        x = torch.cat(parts, dim=0)

        edge_parts: list[torch.Tensor] = []
        rel_parts: list[torch.Tensor] = []
        for rid, etype in enumerate(sorted(data.edge_types)):
            ei = data[etype].edge_index
            if ei.numel() == 0:
                continue
            src_type, _, dst_type = etype
            global_ei = ei.clone()
            global_ei[0] = ei[0] + offsets[src_type]
            global_ei[1] = ei[1] + offsets[dst_type]
            edge_parts.append(global_ei)
            rel_parts.append(
                torch.full(
                    (ei.size(1),),
                    rid,
                    dtype=torch.long,
                    device=ei.device,
                )
            )
        if not edge_parts:
            empty = torch.zeros((2, 0), dtype=torch.long, device=x.device)
            empty_r = torch.zeros((0,), dtype=torch.long, device=x.device)
            return x, empty, empty_r
        edge_index = torch.cat(edge_parts, dim=1)
        edge_type = torch.cat(rel_parts, dim=0)
        return x, edge_index, edge_type

    def _unflatten(self, flat: torch.Tensor, data: Any) -> dict[str, torch.Tensor]:
        offsets, _lookup = global_layout(data)
        sizes = self._type_sizes(data)
        out: dict[str, torch.Tensor] = {}
        for nt in sorted(data.node_types):
            start = offsets[nt]
            out[nt] = flat[start : start + sizes[nt]]
        return out

    def encode(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[Any, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        raise RuntimeError(
            "HeteroRGCNMatchedEncoder.encode(x_dict,...) is unused; call forward(data)."
        )

    def forward(self, data: Any) -> dict[str, torch.Tensor]:
        h_dict = self.input_projection(data.x_dict)
        x, edge_index, edge_type = self._flatten(data, h_dict)
        for conv, norm in zip(self.convs, self.norms):
            prev = x
            x = conv(x, edge_index, edge_type)
            x = torch.relu(x)
            x = self.dropout(x)
            if self.residual:
                x = x + prev
            x = norm(x)
        return self._unflatten(x, data)
