from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, GraphConv, GATv2Conv, TransformerConv
from torch_geometric.nn import JumpingKnowledge
from torch_geometric.nn.norm import PairNorm
from torch_geometric.utils import dropout_edge

from models.TaskB.gnn_common import (
    DEFAULT_TARGET_ENDPOINTS,
    NodeClassificationHead,
    get_targeted_labels,
    get_targeted_wide_scores,
    get_node_labels,
    get_node_mask,
)

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
# Conv typy, dla ktorych dokladamy WLASNY, kontrolowany self_transform w
# _PatientDAGGNNBase.encode(). "gcn" (GraphConv) celowo wykluczone - nie ma
# parametru root_weight, wiec ma WLASNY, nieusuwalny self-transform wbudowany
# w kazde wywolanie warstwy; dolozenie kolejnego by to tylko zdublowalo.
_SELF_TRANSFORM_CONVS = {"sage", "gat", "transformer"}


def _build_conv(conv_type: str, hidden_dim: int, cfg, edge_dim: Optional[int] = None) -> nn.Module:
    conv_type = conv_type.lower()
    if conv_type not in CONV_REGISTRY:
        raise ValueError(f"Unknown conv_type='{conv_type}'. Available: {list(CONV_REGISTRY.keys())}")

    if conv_type == "sage":
        # root_weight=False
        return SAGEConv((hidden_dim, hidden_dim), hidden_dim, root_weight=False)
    if conv_type == "gcn":
        # GraphConv NIE MA parametru root_weight w PyG 
        return GraphConv(hidden_dim, hidden_dim, aggr="add")
    if conv_type == "gat":
        # GATv2Conv (podobnie jak GAT) NIE MA parametru root_weight
        heads = int(getattr(cfg.model, "heads", 4))
        return GATv2Conv(
            (hidden_dim, hidden_dim), hidden_dim // heads, heads=heads,
            add_self_loops=False, edge_dim=edge_dim,
        )
    if conv_type == "transformer":
        # root_weight=False
        heads = int(getattr(cfg.model, "heads", 4))
        return TransformerConv(
            (hidden_dim, hidden_dim), hidden_dim // heads, heads=heads,
            edge_dim=edge_dim, root_weight=False,
        )




def _build_edge_kwargs(conv_type: str, edge_types, data: HeteroData, edge_mask_dict=None):
    edge_weight_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
    edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
    for edge_type in edge_types:
        store = data[edge_type]
        attr = getattr(store, "edge_attr", None)
        if attr is None:
            continue
        if edge_mask_dict is not None and edge_type in edge_mask_dict:
            attr = attr[edge_mask_dict[edge_type]]
        if conv_type in _EDGE_WEIGHT_CONVS:
            # edge_attr ma teraz wiele kolumn: [effect_size*effect_sign, activation_frequency,
            # mechanistic_confidence, evidence_weight, edge_type_onehot...]. Konwolucje
            # przyjmujace skalarny edge_weight (np. GraphConv) dostaja WYLACZNIE pierwsza
            # kolumne (podpisana sila efektu). attr.view(-1) byloby bledem: splaszczyloby
            # [E, F] do wektora dlugosci E*F zamiast E.
            edge_weight_dict[edge_type] = attr[:, 0]
        elif conv_type in _EDGE_ATTR_CONVS:
            edge_attr_dict[edge_type] = attr
    return {"edge_weight_dict": edge_weight_dict, "edge_attr_dict": edge_attr_dict}


# ---------------------------------------------------------------------------
# 0. Wspolna baza: input_proj, embedding tozsamosci wezla, stos HeteroConv
# ---------------------------------------------------------------------------

class _PatientDAGGNNBase(nn.Module):
    """Wspolna baza dla wszystkich architektur GNN na DAGach pacjentow"""

    def __init__(
        self,
        cfg,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        in_dims: Dict[str, int],
        node_names_by_type: Dict[str, List[str]],
        edge_dim: Optional[int] = None,
    ):
        super().__init__()
        hidden_dim = int(cfg.model.hidden_dim)
        num_layers = int(cfg.model.num_layers)
        dropout = float(getattr(cfg.model, "dropout", 0.0))
        aggr = str(getattr(cfg.model, "aggr", "sum"))
        conv_type = str(getattr(cfg.model, "conv_type", "sage")).lower()

        # edge_dim: np. topology["edge_attr_dim"]
        resolved_edge_dim = edge_dim if edge_dim is not None else int(getattr(cfg.model, "edge_dim", 1))

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_type = conv_type
        self.edge_dim = resolved_edge_dim
        self.node_types = list(metadata[0])
        self.edge_types = list(metadata[1])
        self.use_residual = bool(getattr(cfg.model, "use_residual", True))
        self.grad_clip = float(getattr(cfg.training, "grad_clip", 1.0))

        if conv_type == "gcn" and self.use_residual:
            warnings.warn(
                "conv_type='gcn' (GraphConv) nie ma parametru root_weight - "
                "ma wlasny, nieusuwalny self-transform w kazdej warstwie. "
                "use_residual=True jest dla tej architektury ignorowany "
                "(zewnetrzny self_transform nie jest dodawany, zeby nie "
                "zdublowac wbudowanego). Patrz docstring _PatientDAGGNNBase.",
                stacklevel=2,
            )

        self.input_proj = nn.ModuleDict({
            node_type: nn.Linear(int(in_dims[node_type]), hidden_dim)
            for node_type in self.node_types
        })

        # Embedding tozsamosci wezla
        self.node_embedding = nn.ModuleDict({
            node_type: nn.Embedding(len(node_names_by_type[node_type]), hidden_dim)
            for node_type in self.node_types
        })

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {
                edge_type: _build_conv(conv_type, hidden_dim, cfg, edge_dim=resolved_edge_dim)
                for edge_type in self.edge_types
            }
            self.convs.append(HeteroConv(conv_dict, aggr=aggr))

        # Jeden, kontrolowany self-sygnal per warstwa per typ wezla 
        if conv_type in _SELF_TRANSFORM_CONVS:
            self.self_transform: Optional[nn.ModuleList] = nn.ModuleList([
                nn.ModuleDict({nt: nn.Linear(hidden_dim, hidden_dim) for nt in self.node_types})
                for _ in range(num_layers)
            ])
        else:
            self.self_transform = None

        # --- Interwencje anty-oversmoothing (wszystkie domyslnie WYLACZONE,
        # zeby dotychczasowe runy odtwarzaly sie bez zmian) ---

        self.pair_norm_scale = float(getattr(cfg.model, "pair_norm_scale", 0.0))
        self.pair_norm = PairNorm(scale=self.pair_norm_scale) if self.pair_norm_scale > 0 else None
     
        #drop edge
        self.drop_edge = float(getattr(cfg.model, "drop_edge", 0.0))
        #
        # JumpingKnowledge (Xu i in. 2018)
        jk_mode = getattr(cfg.model, "jk_mode", None)
        self.jk_mode = str(jk_mode).lower() if jk_mode else None
        if self.jk_mode in {"none", "null", ""}:
            self.jk_mode = None
        if self.jk_mode is not None:
            self.jk = JumpingKnowledge(
                mode=self.jk_mode, channels=hidden_dim, num_layers=num_layers
            )
            head_in_dim = hidden_dim * num_layers if self.jk_mode == "cat" else hidden_dim
        else:
            self.jk = None
            head_in_dim = hidden_dim

        self.head = NodeClassificationHead(head_in_dim, dropout=dropout)

    def encode(
        self, data: HeteroData, return_layer_reprs: bool = False
    ) -> Dict[str, torch.Tensor] | Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        #Zwraca reprezentacje koncowe per typ wezla.

        x_dict = {
            node_type: F.relu(
                self.input_proj[node_type](data[node_type].x)
                + self.node_embedding[node_type](data[node_type].node_idx)
            )
            for node_type in self.node_types
        }

        # Bez DropEdge topologia jest stala, wiec edge_kwargs jest liczony raz
        use_drop_edge = self.drop_edge > 0.0 and self.training
        static_edge_kwargs = (
            None if use_drop_edge
            else _build_edge_kwargs(self.conv_type, self.edge_types, data)
        )

        layer_reprs: List[Dict[str, torch.Tensor]] = [{
            nt: x.detach() for nt, x in x_dict.items()
        }] if return_layer_reprs else []

        #  dla JumpingKnowledge.
        jk_inputs: Dict[str, List[torch.Tensor]] = {nt: [] for nt in self.node_types}

        for layer_idx, conv in enumerate(self.convs):
            if use_drop_edge:
                edge_index_dict, edge_mask_dict = {}, {}
                for edge_type, edge_index in data.edge_index_dict.items():
                    kept_index, kept_mask = dropout_edge(
                        edge_index, p=self.drop_edge, training=True
                    )
                    edge_index_dict[edge_type] = kept_index
                    edge_mask_dict[edge_type] = kept_mask
                edge_kwargs = _build_edge_kwargs(
                    self.conv_type, self.edge_types, data, edge_mask_dict
                )
            else:
                edge_index_dict = data.edge_index_dict
                edge_kwargs = static_edge_kwargs

            call_kwargs = {}
            if edge_kwargs["edge_weight_dict"]:
                call_kwargs["edge_weight_dict"] = edge_kwargs["edge_weight_dict"]
            if edge_kwargs["edge_attr_dict"]:
                call_kwargs["edge_attr_dict"] = edge_kwargs["edge_attr_dict"]
            x_dict_new = conv(x_dict, edge_index_dict, **call_kwargs)

            updated: Dict[str, torch.Tensor] = {}
            for node_type, current in x_dict.items():
                new_val = x_dict_new.get(node_type)
                if new_val is None:
                    updated[node_type] = current
                    continue

                transformed = F.dropout(F.relu(new_val), p=self.dropout, training=self.training)

                if self.self_transform is None:
                    updated[node_type] = transformed
                elif self.use_residual:
                    self_z = self.self_transform[layer_idx][node_type](current)
                    updated[node_type] = self_z + transformed
                else:
                    updated[node_type] = transformed

            if self.pair_norm is not None:
                # batch: normalizacja MUSI byc per pacjent, nie po calym batchu 
                for node_type in updated:
                    batch_vec = getattr(data[node_type], "batch", None)
                    updated[node_type] = self.pair_norm(updated[node_type], batch_vec)

            x_dict = updated
            for node_type in self.node_types:
                jk_inputs[node_type].append(x_dict[node_type])

            if return_layer_reprs:
                layer_reprs.append({nt: x.detach() for nt, x in x_dict.items()})

        if self.jk is not None:
            x_dict = {nt: self.jk(jk_inputs[nt]) for nt in self.node_types}
            if return_layer_reprs:
                # Ostatni snapshot to reprezentacja PO JK - to ona trafia do
                # glowicy, wiec metryki koncowe powinny dotyczyc wlasnie jej.
                layer_reprs.append({nt: x.detach() for nt, x in x_dict.items()})

        if return_layer_reprs:
            return x_dict, layer_reprs
        return x_dict


# ---------------------------------------------------------------------------
# 1. Baseline: klasyfikuje WSZYSTKIE wezly typu target_node_type
# ---------------------------------------------------------------------------

class SimplePatientDAGNodeClassifier(_PatientDAGGNNBase):

    def __init__(
        self,
        cfg,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        in_dims: Dict[str, int],
        node_names_by_type: Dict[str, List[str]],
        target_node_type: str = "clinical_endpoint",
        edge_dim: Optional[int] = None,
    ):
        super().__init__(
            cfg=cfg, metadata=metadata, in_dims=in_dims,
            node_names_by_type=node_names_by_type, edge_dim=edge_dim,
        )
        self.target_node_type = target_node_type

    def forward(self, data: HeteroData) -> torch.Tensor:
        z_dict = self.encode(data)
        target_z = z_dict[self.target_node_type]
        return self.head(target_z)


# ---------------------------------------------------------------------------
# 2. Targeted: klasyfikuje TYLKO wybrana podliste endpointow
# ---------------------------------------------------------------------------

class TargetedPatientDAGNodeClassifier(_PatientDAGGNNBase):
    """Baseline heterogeniczny GNN do node classification - wersja pod wybrane endpoint, a nie wszystkie oznaczone w DAG"""

    def __init__(
        self,
        cfg,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        in_dims: Dict[str, int],
        node_names_by_type: Dict[str, List[str]],
        target_node_type: str = "clinical_endpoint",
        target_endpoint_names: Optional[List[str]] = None,
        edge_dim: Optional[int] = None,
    ):
        super().__init__(
            cfg=cfg, metadata=metadata, in_dims=in_dims,
            node_names_by_type=node_names_by_type, edge_dim=edge_dim,
        )
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

        self.use_hcr_wide = bool(getattr(cfg.model, "use_hcr_wide", False))
        if self.use_hcr_wide:
            # Inicjalizacja na zero: model startuje identycznie jak bez HCR,
            # trening sam odkrywa, czy sciezka wide cokolwiek wnosi per endpoint.
            self.hcr_wide_weight = nn.Parameter(torch.zeros(len(requested)))
        else:
            self.hcr_wide_weight = None

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
            deep_logits = self._select_targets(logits_all, batch_size, n_per_graph)
        else:
            deep_logits = logits_all[self.target_local_idx]

        if not self.use_hcr_wide:
            return deep_logits
        
        wide = get_targeted_wide_scores(data, self.target_local_idx, self.target_node_type)
        weight = self.hcr_wide_weight.repeat(batch_size) if batch_size > 1 else self.hcr_wide_weight
        return deep_logits + weight * wide

