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

# Conv typy ktore natywnie przyjmuja skalarny `edge_weight`.
_EDGE_WEIGHT_CONVS = {"gcn"}
# Conv typy ktore natywnie przyjmuja (wielowymiarowy) `edge_attr`.
_EDGE_ATTR_CONVS = {"gat", "transformer"}

# Domyslna lista 10 pierwotnych endpointow (mozna nadpisac przez cfg.data.target_endpoints).
DEFAULT_TARGET_ENDPOINTS = [
    "AKI", "DILI", "Depression", "Falls", "Delirium", "GI_bleeding",
    "Hyponatremia", "Hyperkalemia", "QT_arrhythmia", "Hospitalization",
]


def _build_conv(conv_type: str, hidden_dim: int, cfg, edge_dim: Optional[int] = None) -> nn.Module:
    conv_type = conv_type.lower()
    if conv_type not in CONV_REGISTRY:
        raise ValueError(f"Unknown conv_type='{conv_type}'. Available: {list(CONV_REGISTRY.keys())}")

    if conv_type == "sage":
        return SAGEConv((hidden_dim, hidden_dim), hidden_dim)
    if conv_type == "gcn":
        return GraphConv(hidden_dim, hidden_dim, aggr="add")
    if conv_type == "gat":
        heads = int(getattr(cfg.model, "heads", 4))
        return GATv2Conv(
            (hidden_dim, hidden_dim), hidden_dim // heads, heads=heads,
            add_self_loops=False, edge_dim=edge_dim,
        )
    if conv_type == "transformer":
        heads = int(getattr(cfg.model, "heads", 4))
        return TransformerConv(
            (hidden_dim, hidden_dim), hidden_dim // heads, heads=heads, edge_dim=edge_dim,
        )


class NodeClassificationHead(nn.Module):
    """Prosty MLP klasyfikujacy per-wezel."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z).view(-1)


def _build_edge_kwargs(conv_type: str, edge_types, data: HeteroData):
    edge_weight_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
    edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
    for edge_type in edge_types:
        store = data[edge_type]
        attr = getattr(store, "edge_attr", None)
        if attr is None:
            continue
        if conv_type in _EDGE_WEIGHT_CONVS:
            edge_weight_dict[edge_type] = attr.view(-1)
        elif conv_type in _EDGE_ATTR_CONVS:
            edge_attr_dict[edge_type] = attr
    return {"edge_weight_dict": edge_weight_dict, "edge_attr_dict": edge_attr_dict}


# ---------------------------------------------------------------------------
# 1. Baseline: klasyfikuje WSZYSTKIE wezly typu target_node_type
# ---------------------------------------------------------------------------

class SimplePatientDAGNodeClassifier(nn.Module):

    def __init__(
        self,
        cfg,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        in_dims: Dict[str, int],
        target_node_type: str = "clinical_endpoint",
    ):
        super().__init__()
        hidden_dim = int(cfg.model.hidden_dim)
        num_layers = int(cfg.model.num_layers)
        dropout = float(getattr(cfg.model, "dropout", 0.0))
        aggr = str(getattr(cfg.model, "aggr", "sum"))
        conv_type = str(getattr(cfg.model, "conv_type", "sage")).lower()
        edge_dim = int(getattr(cfg.model, "edge_dim", 1))

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_type = conv_type
        self.edge_dim = edge_dim
        self.node_types = list(metadata[0])
        self.edge_types = list(metadata[1])
        self.target_node_type = target_node_type

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

        self.head = NodeClassificationHead(hidden_dim, dropout=dropout)

    def encode(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        x_dict = {
            node_type: F.relu(self.input_proj[node_type](data[node_type].x))
            for node_type in self.node_types
        }
        edge_kwargs = _build_edge_kwargs(self.conv_type, self.edge_types, data)

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

    def forward(self, data: HeteroData) -> torch.Tensor:
        z_dict = self.encode(data)
        target_z = z_dict[self.target_node_type]
        return self.head(target_z)


# ---------------------------------------------------------------------------
# 2. Targeted: klasyfikuje TYLKO wybrana podliste endpointow
# ---------------------------------------------------------------------------

class TargetedPatientDAGNodeClassifier(nn.Module):
    """Baseline heterogeniczny GNN do node classification, z jawnym
    ograniczeniem predykcji do wybranej podlisty wezlow-endpointow
    (target_endpoint_names), w stalej kolejnosci."""

    def __init__(
        self,
        cfg,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        in_dims: Dict[str, int],
        node_names_by_type: Dict[str, List[str]],
        target_node_type: str = "clinical_endpoint",
        target_endpoint_names: Optional[List[str]] = None,
    ):
        super().__init__()
        hidden_dim = int(cfg.model.hidden_dim)
        num_layers = int(cfg.model.num_layers)
        dropout = float(getattr(cfg.model, "dropout", 0.0))
        aggr = str(getattr(cfg.model, "aggr", "sum"))
        conv_type = str(getattr(cfg.model, "conv_type", "sage")).lower()
        edge_dim = int(getattr(cfg.model, "edge_dim", 1))

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_type = conv_type
        self.edge_dim = edge_dim
        self.node_types = list(metadata[0])
        self.edge_types = list(metadata[1])
        self.target_node_type = target_node_type

        all_endpoint_names = node_names_by_type[target_node_type]
        requested = target_endpoint_names or getattr(cfg.data, "target_endpoints", None) or DEFAULT_TARGET_ENDPOINTS
        requested = list(requested)

        missing = [e for e in requested if e not in all_endpoint_names]
        if missing:
            raise ValueError(
                f"target_endpoint_names {missing} nie sa wezlami typu "
                f"'{target_node_type}' w DAG. Dostepne: {all_endpoint_names}"
            )

        self.target_endpoint_names = requested
        name_to_local_idx = {name: i for i, name in enumerate(all_endpoint_names)}
        self.register_buffer(
            "target_local_idx",
            torch.tensor([name_to_local_idx[e] for e in requested], dtype=torch.long),
        )

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

        self.head = NodeClassificationHead(hidden_dim, dropout=dropout)

    def encode(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        x_dict = {
            node_type: F.relu(self.input_proj[node_type](data[node_type].x))
            for node_type in self.node_types
        }
        edge_kwargs = _build_edge_kwargs(self.conv_type, self.edge_types, data)

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

    def _select_targets(self, full_tensor: torch.Tensor, batch_size: int, n_endpoint_nodes_per_graph: int) -> torch.Tensor:
        offsets = torch.arange(batch_size, device=full_tensor.device) * n_endpoint_nodes_per_graph
        idx = (offsets.unsqueeze(1) + self.target_local_idx.unsqueeze(0)).view(-1)
        return full_tensor[idx]

    def forward(self, data: HeteroData) -> torch.Tensor:
        z_dict = self.encode(data)
        target_z = z_dict[self.target_node_type]
        logits_all = self.head(target_z)

        batch_size = int(data[self.target_node_type].batch.max().item()) + 1 \
            if hasattr(data[self.target_node_type], "batch") else 1
        n_per_graph = target_z.size(0) // max(batch_size, 1)

        if batch_size > 1:
            return self._select_targets(logits_all, batch_size, n_per_graph)
        return logits_all[self.target_local_idx]


def get_targeted_labels(
    data: HeteroData,
    target_local_idx: torch.Tensor,
    target_node_type: str = "clinical_endpoint",
) -> torch.Tensor:
    """Etykiety odfiltrowane do target_endpoint_names, w tej samej
    kolejnosci co logity z forward()."""
    y = data[target_node_type].y.float()
    batch_size = int(data[target_node_type].batch.max().item()) + 1 \
        if hasattr(data[target_node_type], "batch") else 1
    n_per_graph = y.size(0) // max(batch_size, 1)
    offsets = torch.arange(batch_size, device=y.device) * n_per_graph
    idx = (offsets.unsqueeze(1) + target_local_idx.unsqueeze(0)).view(-1)
    return y[idx]


def get_node_labels(data: HeteroData, target_node_type: str = "clinical_endpoint") -> torch.Tensor:
    """Etykiety dla WSZYSTKICH wezlow target_node_type (uzyj z
    SimplePatientDAGNodeClassifier, nie z wariantem targeted)."""
    return data[target_node_type].y.float()


def get_node_mask(data: HeteroData, target_node_type: str = "clinical_endpoint") -> torch.Tensor:
    """Maska wskazujaca, ktore wezly maja rzeczywista etykiete."""
    return data[target_node_type].y_mask
