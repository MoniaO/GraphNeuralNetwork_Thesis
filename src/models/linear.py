from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import nn
from torch_geometric.data import HeteroData


class SimpleEdgeLinear(nn.Module):
    def __init__(self, in_dim_patient: int, in_dim_variable: int):
        super().__init__()
        self.linear = nn.Linear(in_dim_patient + in_dim_variable, 1)

    def forward(self, patient_x: torch.Tensor, variable_x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([patient_x, variable_x], dim=-1)
        return self.linear(x).view(-1)


class LinearHeteroLP(nn.Module):
    def __init__(
        self,
        cfg,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        in_dims: Dict[str, int],
    ):
        super().__init__()
        self.node_types = list(metadata[0])
        self.edge_types = list(metadata[1])

        self.patient_dim = int(in_dims["patient"])
        self.variable_dim = int(in_dims["variable"])

        self.decoder = SimpleEdgeLinear(
            in_dim_patient=self.patient_dim,
            in_dim_variable=self.variable_dim,
        )

    def encode(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        return {
            "patient": data["patient"].x,
            "variable": data["variable"].x,
        }

    def decode(self, z_dict: Dict[str, torch.Tensor], edge_label_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_label_index
        src_x = z_dict["patient"][src]
        dst_x = z_dict["variable"][dst]
        return self.decoder(src_x, dst_x)

    def forward(self, data: HeteroData) -> torch.Tensor:
        z_dict = self.encode(data)
        edge_label_index = data[("patient", "has_adr", "variable")].edge_label_index
        return self.decode(z_dict, edge_label_index)