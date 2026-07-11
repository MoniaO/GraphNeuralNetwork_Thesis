from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv


class SimpleEdgeDecoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.lin = nn.Linear(hidden_dim * 2, 1)

    def forward(self, src_z: torch.Tensor, dst_z: torch.Tensor) -> torch.Tensor:
        pair = torch.cat([src_z, dst_z], dim=-1)
        return self.lin(pair).view(-1)


class SimpleHeteroGNN(nn.Module):
    def __init__(self, cfg, metadata: Tuple[List[str], List[Tuple[str, str, str]]], in_dims: Dict[str, int]):
        super().__init__()
        hidden_dim = int(cfg.model.hidden_dim)
        num_layers = int(cfg.model.num_layers)
        aggr = str(getattr(cfg.model, "aggr", "sum"))

        self.node_types = list(metadata[0])
        self.edge_types = list(metadata[1])

        self.input_proj = nn.ModuleDict({
            node_type: nn.Linear(in_dims[node_type], hidden_dim)
            for node_type in self.node_types
        })

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: SAGEConv((-1, -1), hidden_dim)
                    for edge_type in self.edge_types
                },
                aggr=aggr,
            )
            self.convs.append(conv)

        self.decoder = SimpleEdgeDecoder(hidden_dim)

    def encode(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        x_dict = {
            node_type: F.relu(self.input_proj[node_type](data[node_type].x))
            for node_type in self.node_types
        }

        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {node_type: F.relu(x) for node_type, x in x_dict.items()}

        return x_dict

    def decode(self, z_dict: Dict[str, torch.Tensor], edge_label_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_label_index
        src_z = z_dict["patient"][src]
        dst_z = z_dict["variable"][dst]
        return self.decoder(src_z, dst_z)

    def forward(self, data: HeteroData) -> torch.Tensor:
        z_dict = self.encode(data)
        edge_label_index = data[("patient", "has_adr", "variable")].edge_label_index
        return self.decode(z_dict, edge_label_index)
