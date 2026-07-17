
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, GraphConv, GATv2Conv, TransformerConv

CONV_REGISTRY = {
    "sage": SAGEConv,
    "gcn": GraphConv,
    "gat": GATv2Conv,
    "transformer": TransformerConv,
}

# Conv types that natively accept a scalar `edge_weight` kwarg.
_EDGE_WEIGHT_CONVS = {"gcn"}
# Conv types that natively accept a (possibly multi-dim) `edge_attr` kwarg.
_EDGE_ATTR_CONVS = {"gat", "transformer"}


def _build_conv(conv_type: str, hidden_dim: int, cfg, edge_dim: Optional[int] = None) -> nn.Module:
    conv_type = conv_type.lower()
    if conv_type not in CONV_REGISTRY:
        raise ValueError(
            f"Unknown conv_type='{conv_type}'. Available: {list(CONV_REGISTRY.keys())}"
        )

    if conv_type == "sage":
        # SAGEConv has no native edge-feature support. If effect_size / value
        # matter for this relation, switch to 'gat' or 'transformer'.
        return SAGEConv((hidden_dim, hidden_dim), hidden_dim)

    if conv_type == "gcn":
        # GraphConv supports a scalar `edge_weight` per edge (used to carry
        # effect_size / value). add_self_loops kept False for bipartite edges.
        return GraphConv(hidden_dim, hidden_dim, aggr="add")

    if conv_type == "gat":
        heads = int(getattr(cfg.model, "heads", 4))
        return GATv2Conv(
            (hidden_dim, hidden_dim),
            hidden_dim // heads,
            heads=heads,
            add_self_loops=False,
            edge_dim=edge_dim,
        )

    if conv_type == "transformer":
        heads = int(getattr(cfg.model, "heads", 4))
        return TransformerConv(
            (hidden_dim, hidden_dim),
            hidden_dim // heads,
            heads=heads,
            edge_dim=edge_dim,
        )


class EdgeDecoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, src_z: torch.Tensor, dst_z: torch.Tensor) -> torch.Tensor:
        pair_z = torch.cat([src_z, dst_z], dim=-1)
        return self.mlp(pair_z).view(-1)


class SimpleHeteroGNN(nn.Module):
    def __init__(
        self,
        cfg,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        in_dims: Dict[str, int],
    ):
        super().__init__()
        hidden_dim = int(cfg.model.hidden_dim)
        num_layers = int(cfg.model.num_layers)
        dropout = float(getattr(cfg.model, "dropout", 0.0))
        aggr = str(getattr(cfg.model, "aggr", "sum"))
        conv_type = str(getattr(cfg.model, "conv_type", "sage")).lower()
        # effect_size / value are stored as single-column tensors -> dim 1.
        edge_dim = int(getattr(cfg.model, "edge_dim", 1))

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_type = conv_type
        self.edge_dim = edge_dim
        self.node_types = list(metadata[0])
        self.edge_types = list(metadata[1])

        self.input_proj = nn.ModuleDict({
            node_type: nn.Linear(int(in_dims[node_type]), hidden_dim)
            for node_type in self.node_types
        })

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {
                edge_type: _build_conv(conv_type, hidden_dim, cfg, edge_dim=edge_dim)
                for edge_type in self.edge_types
            }
            self.convs.append(HeteroConv(conv_dict, aggr=aggr))

        self.decoder = EdgeDecoder(hidden_dim)

    def _build_edge_kwargs(self, data: HeteroData) -> Dict[str, Dict[Tuple[str, str, str], torch.Tensor]]:
        """Route each relation's edge_attr to the kwarg its conv layer supports.

        - GraphConv ("gcn") expects a scalar `edge_weight` per edge.
        - GATv2Conv / TransformerConv ("gat"/"transformer") expect `edge_attr`.
        - SAGEConv ("sage") has no edge-feature support -> silently skipped.
        """
        edge_weight_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
        edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}

        for edge_type in self.edge_types:
            store = data[edge_type]
            attr = getattr(store, "edge_attr", None)
            if attr is None:
                continue

            if self.conv_type in _EDGE_WEIGHT_CONVS:
                edge_weight_dict[edge_type] = attr.view(-1)
            elif self.conv_type in _EDGE_ATTR_CONVS:
                edge_attr_dict[edge_type] = attr

        return {"edge_weight_dict": edge_weight_dict, "edge_attr_dict": edge_attr_dict}

    def encode(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        x_dict = {
            node_type: F.relu(self.input_proj[node_type](data[node_type].x))
            for node_type in self.node_types
        }

        edge_kwargs = self._build_edge_kwargs(data)

        for conv in self.convs:
            call_kwargs = {}
            if edge_kwargs["edge_weight_dict"]:
                call_kwargs["edge_weight_dict"] = edge_kwargs["edge_weight_dict"]
            if edge_kwargs["edge_attr_dict"]:
                call_kwargs["edge_attr_dict"] = edge_kwargs["edge_attr_dict"]

            x_dict = conv(x_dict, data.edge_index_dict, **call_kwargs)
            x_dict = {
                node_type: F.dropout(F.relu(x), p=self.dropout, training=self.training)
                for node_type, x in x_dict.items()
            }

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
