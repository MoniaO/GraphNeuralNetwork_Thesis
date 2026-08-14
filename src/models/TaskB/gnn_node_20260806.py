# from __future__ import annotations

# import warnings
# from typing import Dict, List, Optional, Tuple

# import torch
# import torch.nn.functional as F
# from torch import nn
# from torch_geometric.data import HeteroData
# from torch_geometric.nn import HeteroConv, SAGEConv, GraphConv, GATv2Conv, TransformerConv
# from torch_geometric.nn import JumpingKnowledge
# from torch_geometric.nn.norm import PairNorm
# from torch_geometric.utils import dropout_edge

# from models.TaskB.gnn_common import (
#     DEFAULT_TARGET_ENDPOINTS,
#     NodeClassificationHead,
#     get_targeted_labels,
#     get_targeted_wide_scores,
#     get_node_labels,
#     get_node_mask,
# )

# CONV_REGISTRY = {
#     "sage": SAGEConv,
#     "gcn": GraphConv,
#     "gat": GATv2Conv,
#     "transformer": TransformerConv,
# }

# # Conv typy ktore natywnie przyjmuja skalarny `edge_weight`.
# _EDGE_WEIGHT_CONVS = {"gcn"}
# # Conv typy ktore natywnie przyjmuja (wielowymiarowy) `edge_attr`.
# _EDGE_ATTR_CONVS = {"gat", "transformer"}
# # Conv typy, dla ktorych dokladamy WLASNY, kontrolowany self_transform w
# # _PatientDAGGNNBase.encode(). "gcn" (GraphConv) celowo wykluczone - nie ma
# # parametru root_weight, wiec ma WLASNY, nieusuwalny self-transform wbudowany
# # w kazde wywolanie warstwy; dolozenie kolejnego by to tylko zdublowalo.
# _SELF_TRANSFORM_CONVS = {"sage", "gat", "transformer"}


# def _build_conv(conv_type: str, hidden_dim: int, cfg, edge_dim: Optional[int] = None) -> nn.Module:
#     conv_type = conv_type.lower()
#     if conv_type not in CONV_REGISTRY:
#         raise ValueError(f"Unknown conv_type='{conv_type}'. Available: {list(CONV_REGISTRY.keys())}")

#     if conv_type == "sage":
#         # root_weight=False
#         return SAGEConv((hidden_dim, hidden_dim), hidden_dim, root_weight=False)
#     if conv_type == "gcn":
#         # GraphConv NIE MA parametru root_weight w PyG 
#         return GraphConv(hidden_dim, hidden_dim, aggr="add")
#     if conv_type == "gat":
#         # GATv2Conv (podobnie jak GAT) NIE MA parametru root_weight
#         heads = int(getattr(cfg.model, "heads", 4))
#         return GATv2Conv(
#             (hidden_dim, hidden_dim), hidden_dim // heads, heads=heads,
#             add_self_loops=False, edge_dim=edge_dim,
#         )
#     if conv_type == "transformer":
#         # root_weight=False
#         heads = int(getattr(cfg.model, "heads", 4))
#         return TransformerConv(
#             (hidden_dim, hidden_dim), hidden_dim // heads, heads=heads,
#             edge_dim=edge_dim, root_weight=False,
#         )




# def _build_edge_kwargs(conv_type: str, edge_types, data: HeteroData, edge_mask_dict=None):
#     edge_weight_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
#     edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
#     for edge_type in edge_types:
#         store = data[edge_type]
#         attr = getattr(store, "edge_attr", None)
#         if attr is None:
#             continue
#         if edge_mask_dict is not None and edge_type in edge_mask_dict:
#             attr = attr[edge_mask_dict[edge_type]]
#         if conv_type in _EDGE_WEIGHT_CONVS:
#             # edge_attr ma teraz wiele kolumn: [effect_size*effect_sign, activation_frequency,
#             # mechanistic_confidence, evidence_weight, edge_type_onehot...]. Konwolucje
#             # przyjmujace skalarny edge_weight (np. GraphConv) dostaja WYLACZNIE pierwsza
#             # kolumne (podpisana sila efektu). attr.view(-1) byloby bledem: splaszczyloby
#             # [E, F] do wektora dlugosci E*F zamiast E.
#             edge_weight_dict[edge_type] = attr[:, 0]
#         elif conv_type in _EDGE_ATTR_CONVS:
#             edge_attr_dict[edge_type] = attr
#     return {"edge_weight_dict": edge_weight_dict, "edge_attr_dict": edge_attr_dict}


# # ---------------------------------------------------------------------------
# # 0. Wspolna baza: input_proj, embedding tozsamosci wezla, stos HeteroConv
# # ---------------------------------------------------------------------------

# class _PatientDAGGNNBase(nn.Module):
#     """Wspolna baza dla wszystkich architektur GNN na DAGach pacjentow"""

#     def __init__(
#         self,
#         cfg,
#         metadata: Tuple[List[str], List[Tuple[str, str, str]]],
#         in_dims: Dict[str, int],
#         node_names_by_type: Dict[str, List[str]],
#         edge_dim: Optional[int] = None,
#     ):
#         super().__init__()
#         hidden_dim = int(cfg.model.hidden_dim)
#         num_layers = int(cfg.model.num_layers)
#         dropout = float(getattr(cfg.model, "dropout", 0.0))
#         aggr = str(getattr(cfg.model, "aggr", "sum"))
#         conv_type = str(getattr(cfg.model, "conv_type", "sage")).lower()

#         # edge_dim: np. topology["edge_attr_dim"]
#         resolved_edge_dim = edge_dim if edge_dim is not None else int(getattr(cfg.model, "edge_dim", 1))

#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#         self.dropout = dropout
#         self.conv_type = conv_type
#         self.edge_dim = resolved_edge_dim
#         self.node_types = list(metadata[0])
#         self.edge_types = list(metadata[1])
#         self.use_residual = bool(getattr(cfg.model, "use_residual", True))
#         self.grad_clip = float(getattr(cfg.training, "grad_clip", 1.0))

#         if conv_type == "gcn" and self.use_residual:
#             warnings.warn(
#                 "conv_type='gcn' (GraphConv) nie ma parametru root_weight - "
#                 "ma wlasny, nieusuwalny self-transform w kazdej warstwie. "
#                 "use_residual=True jest dla tej architektury ignorowany "
# #                 "(zewnetrzny self_transform nie jest dodawany, zeby nie "
# #                 "zdublowac wbudowanego). Patrz docstring _PatientDAGGNNBase.",
# #                 stacklevel=2,
# #             )

# #         self.input_proj = nn.ModuleDict({
# #             node_type: nn.Linear(int(in_dims[node_type]), hidden_dim)
# #             for node_type in self.node_types
# #         })

# #         # Embedding tozsamosci wezla
# #         self.node_embedding = nn.ModuleDict({
# #             node_type: nn.Embedding(len(node_names_by_type[node_type]), hidden_dim)
# #             for node_type in self.node_types
# #         })

# #         self.convs = nn.ModuleList()
# #         for _ in range(num_layers):
# #             conv_dict = {
# #                 edge_type: _build_conv(conv_type, hidden_dim, cfg, edge_dim=resolved_edge_dim)
# #                 for edge_type in self.edge_types
# #             }
# #             self.convs.append(HeteroConv(conv_dict, aggr=aggr))

# #         # Jeden, kontrolowany self-sygnal per warstwa per typ wezla 
# #         if conv_type in _SELF_TRANSFORM_CONVS:
# #             self.self_transform: Optional[nn.ModuleList] = nn.ModuleList([
# #                 nn.ModuleDict({nt: nn.Linear(hidden_dim, hidden_dim) for nt in self.node_types})
# #                 for _ in range(num_layers)
# #             ])
# #         else:
# #             self.self_transform = None

# #         # --- Interwencje anty-oversmoothing (wszystkie domyslnie WYLACZONE,
# #         # zeby dotychczasowe runy odtwarzaly sie bez zmian) ---

# #         self.pair_norm_scale = float(getattr(cfg.model, "pair_norm_scale", 0.0))
# #         self.pair_norm = PairNorm(scale=self.pair_norm_scale) if self.pair_norm_scale > 0 else None
     
# #         #drop edge
# #         self.drop_edge = float(getattr(cfg.model, "drop_edge", 0.0))
# #         #
# #         # JumpingKnowledge (Xu i in. 2018)
# #         jk_mode = getattr(cfg.model, "jk_mode", None)
# #         self.jk_mode = str(jk_mode).lower() if jk_mode else None
# #         if self.jk_mode in {"none", "null", ""}:
# #             self.jk_mode = None
# #         if self.jk_mode is not None:
# #             self.jk = JumpingKnowledge(
# #                 mode=self.jk_mode, channels=hidden_dim, num_layers=num_layers
# #             )
# #             head_in_dim = hidden_dim * num_layers if self.jk_mode == "cat" else hidden_dim
# #         else:
# #             self.jk = None
# #             head_in_dim = hidden_dim

# #         self.head = NodeClassificationHead(head_in_dim, dropout=dropout)

# #     def encode(
# #         self, data: HeteroData, return_layer_reprs: bool = False
# #     ) -> Dict[str, torch.Tensor] | Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
# #         #Zwraca reprezentacje koncowe per typ wezla.

# #         x_dict = {
# #             node_type: F.relu(
# #                 self.input_proj[node_type](data[node_type].x)
# #                 + self.node_embedding[node_type](data[node_type].node_idx)
# #             )
# #             for node_type in self.node_types
# #         }

# #         # Bez DropEdge topologia jest stala, wiec edge_kwargs jest liczony raz
# #         use_drop_edge = self.drop_edge > 0.0 and self.training
# #         static_edge_kwargs = (
# #             None if use_drop_edge
# #             else _build_edge_kwargs(self.conv_type, self.edge_types, data)
# #         )

# #         layer_reprs: List[Dict[str, torch.Tensor]] = [{
# #             nt: x.detach() for nt, x in x_dict.items()
# #         }] if return_layer_reprs else []

# #         #  dla JumpingKnowledge.
# #         jk_inputs: Dict[str, List[torch.Tensor]] = {nt: [] for nt in self.node_types}

# #         for layer_idx, conv in enumerate(self.convs):
# #             if use_drop_edge:
# #                 edge_index_dict, edge_mask_dict = {}, {}
# #                 for edge_type, edge_index in data.edge_index_dict.items():
# #                     kept_index, kept_mask = dropout_edge(
# #                         edge_index, p=self.drop_edge, training=True
# #                     )
# #                     edge_index_dict[edge_type] = kept_index
# #                     edge_mask_dict[edge_type] = kept_mask
# #                 edge_kwargs = _build_edge_kwargs(
# #                     self.conv_type, self.edge_types, data, edge_mask_dict
# #                 )
# #             else:
# #                 edge_index_dict = data.edge_index_dict
# #                 edge_kwargs = static_edge_kwargs

# #             call_kwargs = {}
# #             if edge_kwargs["edge_weight_dict"]:
# #                 call_kwargs["edge_weight_dict"] = edge_kwargs["edge_weight_dict"]
# #             if edge_kwargs["edge_attr_dict"]:
# #                 call_kwargs["edge_attr_dict"] = edge_kwargs["edge_attr_dict"]
# #             x_dict_new = conv(x_dict, edge_index_dict, **call_kwargs)

# #             updated: Dict[str, torch.Tensor] = {}
# #             for node_type, current in x_dict.items():
# #                 new_val = x_dict_new.get(node_type)
# #                 if new_val is None:
# #                     updated[node_type] = current
# #                     continue

# #                 transformed = F.dropout(F.relu(new_val), p=self.dropout, training=self.training)

# #                 if self.self_transform is None:
# #                     updated[node_type] = transformed
# #                 elif self.use_residual:
# #                     self_z = self.self_transform[layer_idx][node_type](current)
# #                     updated[node_type] = self_z + transformed
# #                 else:
# #                     updated[node_type] = transformed

# #             if self.pair_norm is not None:
# #                 # batch: normalizacja MUSI byc per pacjent, nie po calym batchu 
# #                 for node_type in updated:
# #                     batch_vec = getattr(data[node_type], "batch", None)
# #                     updated[node_type] = self.pair_norm(updated[node_type], batch_vec)

# #             x_dict = updated
# #             for node_type in self.node_types:
# #                 jk_inputs[node_type].append(x_dict[node_type])

# #             if return_layer_reprs:
# #                 layer_reprs.append({nt: x.detach() for nt, x in x_dict.items()})

# #         if self.jk is not None:
# #             x_dict = {nt: self.jk(jk_inputs[nt]) for nt in self.node_types}
# #             if return_layer_reprs:
# #                 # Ostatni snapshot to reprezentacja PO JK - to ona trafia do
# #                 # glowicy, wiec metryki koncowe powinny dotyczyc wlasnie jej.
# #                 layer_reprs.append({nt: x.detach() for nt, x in x_dict.items()})

# #         if return_layer_reprs:
# #             return x_dict, layer_reprs
# #         return x_dict


# # # ---------------------------------------------------------------------------
# # # 1. Baseline: klasyfikuje WSZYSTKIE wezly typu target_node_type
# # # ---------------------------------------------------------------------------

# # class SimplePatientDAGNodeClassifier(_PatientDAGGNNBase):

# #     def __init__(
# #         self,
# #         cfg,
# #         metadata: Tuple[List[str], List[Tuple[str, str, str]]],
# #         in_dims: Dict[str, int],
# #         node_names_by_type: Dict[str, List[str]],
# #         target_node_type: str = "clinical_endpoint",
# #         edge_dim: Optional[int] = None,
# #     ):
# #         super().__init__(
# #             cfg=cfg, metadata=metadata, in_dims=in_dims,
# #             node_names_by_type=node_names_by_type, edge_dim=edge_dim,
# #         )
# #         self.target_node_type = target_node_type

# #     def forward(self, data: HeteroData) -> torch.Tensor:
# #         z_dict = self.encode(data)
# #         target_z = z_dict[self.target_node_type]
# #         return self.head(target_z)


# # # ---------------------------------------------------------------------------
# # # 2. Targeted: klasyfikuje TYLKO wybrana podliste endpointow
# # # ---------------------------------------------------------------------------

# # class TargetedPatientDAGNodeClassifier(_PatientDAGGNNBase):
# #     """Baseline heterogeniczny GNN do node classification - wersja pod wybrane endpoint, a nie wszystkie oznaczone w DAG"""

# #     def __init__(
# #         self,
# #         cfg,
# #         metadata: Tuple[List[str], List[Tuple[str, str, str]]],
# #         in_dims: Dict[str, int],
# #         node_names_by_type: Dict[str, List[str]],
# #         target_node_type: str = "clinical_endpoint",
# #         target_endpoint_names: Optional[List[str]] = None,
# #         edge_dim: Optional[int] = None,
# #     ):
# #         super().__init__(
# #             cfg=cfg, metadata=metadata, in_dims=in_dims,
# #             node_names_by_type=node_names_by_type, edge_dim=edge_dim,
# #         )
# #         self.target_node_type = target_node_type

# #         all_endpoint_names = node_names_by_type[target_node_type]
# #         requested = target_endpoint_names or getattr(cfg.data, "target_endpoints", None) or DEFAULT_TARGET_ENDPOINTS
# #         requested = list(requested)

# #         missing = [e for e in requested if e not in all_endpoint_names]
# #         if missing:
# #             raise ValueError(
# #                 f"target_endpoint_names {missing} nie sa wezlami typu "
# #                 f"'{target_node_type}' w DAG. Dostepne: {all_endpoint_names}"
# #             )

# #         self.target_endpoint_names = requested
# #         name_to_local_idx = {name: i for i, name in enumerate(all_endpoint_names)}
# #         self.register_buffer(
# #             "target_local_idx",
# #             torch.tensor([name_to_local_idx[e] for e in requested], dtype=torch.long),
# #         )

# #         self.use_hcr_wide = bool(getattr(cfg.model, "use_hcr_wide", False))
# #         if self.use_hcr_wide:
# #             # Inicjalizacja na zero: model startuje identycznie jak bez HCR,
# #             # trening sam odkrywa, czy sciezka wide cokolwiek wnosi per endpoint.
# #             self.hcr_wide_weight = nn.Parameter(torch.zeros(len(requested)))
# #         else:
# #             self.hcr_wide_weight = None

# #     def _select_targets(self, full_tensor: torch.Tensor, batch_size: int, n_endpoint_nodes_per_graph: int) -> torch.Tensor:
# #         offsets = torch.arange(batch_size, device=full_tensor.device) * n_endpoint_nodes_per_graph
# #         idx = (offsets.unsqueeze(1) + self.target_local_idx.unsqueeze(0)).view(-1)
# #         return full_tensor[idx]

# #     def forward(self, data: HeteroData) -> torch.Tensor:
# #         z_dict = self.encode(data)
# #         target_z = z_dict[self.target_node_type]
# #         logits_all = self.head(target_z)

# #         batch_size = int(data[self.target_node_type].batch.max().item()) + 1 \
# #             if hasattr(data[self.target_node_type], "batch") else 1
# #         n_per_graph = target_z.size(0) // max(batch_size, 1)

# #         if batch_size > 1:
# #             deep_logits = self._select_targets(logits_all, batch_size, n_per_graph)
# #         else:
# #             deep_logits = logits_all[self.target_local_idx]

# #         if not self.use_hcr_wide:
# #             return deep_logits
        
# #         wide = get_targeted_wide_scores(data, self.target_local_idx, self.target_node_type)
# #         weight = self.hcr_wide_weight.repeat(batch_size) if batch_size > 1 else self.hcr_wide_weight
# #         return deep_logits + weight * wide


# from __future__ import annotations

# import warnings
# from typing import Dict, List, Optional, Tuple

# import torch
# import torch.nn.functional as F
# from torch import nn
# from torch_geometric.data import HeteroData
# from torch_geometric.nn import HeteroConv, SAGEConv, GraphConv, GATv2Conv, TransformerConv
# from torch_geometric.nn import JumpingKnowledge
# from torch_geometric.nn.norm import PairNorm
# from torch_geometric.utils import dropout_edge

# from models.TaskB.gnn_common import (
#     DEFAULT_TARGET_ENDPOINTS,
#     NodeClassificationHead,
#     HCRPairEncoder,
#     get_targeted_labels,
#     get_targeted_wide_scores,
#     get_targeted_pair_evidence,
#     get_node_labels,
#     get_node_mask,
# )
# from src.data.PreprocessingTaskB.hcr_wide_features import N_PAIR_EVIDENCE_CHANNELS

# CONV_REGISTRY = {
#     "sage": SAGEConv,
#     "gcn": GraphConv,
#     "gat": GATv2Conv,
#     "transformer": TransformerConv,
# }

# # Conv typy ktore natywnie przyjmuja skalarny `edge_weight`.
# _EDGE_WEIGHT_CONVS = {"gcn"}
# # Conv typy ktore natywnie przyjmuja (wielowymiarowy) `edge_attr`.
# _EDGE_ATTR_CONVS = {"gat", "transformer"}
# # Conv typy, dla ktorych dokladamy WLASNY, kontrolowany self_transform w
# # _PatientDAGGNNBase.encode(). "gcn" (GraphConv) celowo wykluczone - nie ma
# # parametru root_weight, wiec ma WLASNY, nieusuwalny self-transform wbudowany
# # w kazde wywolanie warstwy; dolozenie kolejnego by to tylko zdublowalo.
# _SELF_TRANSFORM_CONVS = {"sage", "gat", "transformer"}


# def _build_conv(conv_type: str, hidden_dim: int, cfg, edge_dim: Optional[int] = None) -> nn.Module:
#     conv_type = conv_type.lower()
#     if conv_type not in CONV_REGISTRY:
#         raise ValueError(f"Unknown conv_type='{conv_type}'. Available: {list(CONV_REGISTRY.keys())}")

#     if conv_type == "sage":
#         # root_weight=False: SAGEConv domyslnie dodaje WLASNY, wewnetrzny
#         # self-transform (W_root @ x_dst) do kazdej wiadomosci. Owiniete w
#         # HeteroConv per-relacja, to sie duplikuje R razy (R = liczba relacji
#         # wchodzacych do danego typu wezla) - kazda relacja ma WLASNA macierz
#         # W_root. root_weight=False usuwa to calkowicie; jedyny self-sygnal
#         # pochodzi teraz z self_transform w _PatientDAGGNNBase (jeden,
#         # kontrolowany, per warstwa per typ wezla - patrz encode()).
#         return SAGEConv((hidden_dim, hidden_dim), hidden_dim, root_weight=False)
#     if conv_type == "gcn":
#         # GraphConv NIE MA parametru root_weight w PyG - self-transform
#         # (lin_root) jest wbudowany bezwarunkowo, bez mozliwosci wylaczenia.
#         # Duplikacja R-krotna (per relacja w HeteroConv) jest wiec dla tej
#         # architektury NIEUSUWALNA bez pisania wlasnej warstwy od zera.
#         # encode() NIE dodaje dla "gcn" zadnego dodatkowego self_transform
#         # (dublowaloby to jeszcze bardziej) - use_residual jest dla "gcn"
#         # ignorowany, z ostrzezeniem przy konstrukcji modelu.
#         return GraphConv(hidden_dim, hidden_dim, aggr="add")
#     if conv_type == "gat":
#         # GATv2Conv (podobnie jak GAT) NIE MA parametru root_weight. Self-
#         # zaleznosc wchodzi inna droga: wektor zapytania (query) w mechanizmie
#         # uwagi jest liczony z x_dst, wiec cechy wezla docelowego ksztaltuja
#         # WAGI uwagi dla sasiadow, nawet bez add_self_loops i bez osobnego
#         # skladnika addytywnego. To NIE jest to samo zjawisko co w SAGEConv
#         # (tam self byl DODAWANY wprost, tu tylko WPLYWA na wagi agregacji) -
#         # nie da sie tego wylaczyc parametrem konstruktora. self_transform w
#         # encode() dziala tu wiec jako DODATKOWY, kontrolowany self-sygnal
#         # NA WIERZCHU tej wbudowanej, nieusuwalnej zaleznosci - use_residual
#         # steruje tylko ta dodatkowa czescia, nie eliminuje zaleznosci
#         # bazowej. Patrz docstring _PatientDAGGNNBase.
#         heads = int(getattr(cfg.model, "heads", 4))
#         return GATv2Conv(
#             (hidden_dim, hidden_dim), hidden_dim // heads, heads=heads,
#             add_self_loops=False, edge_dim=edge_dim,
#         )
#     if conv_type == "transformer":
#         # root_weight=False: TransformerConv ma ten sam addytywny self-skip
#         # co SAGEConv (mozliwy do wylaczenia), ALE dodatkowo, tak jak GAT,
#         # liczy zapytanie w mechanizmie uwagi z x_dst - ta czesc zostaje
#         # nieusuwalna niezaleznie od root_weight. root_weight=False usuwa
#         # WIEC TYLKO addytywny skip, nie cala self-zaleznosc - analogiczne
#         # zastrzezenie jak dla "gat" powyzej.
#         heads = int(getattr(cfg.model, "heads", 4))
#         return TransformerConv(
#             (hidden_dim, hidden_dim), hidden_dim // heads, heads=heads,
#             edge_dim=edge_dim, root_weight=False,
#         )




# def _build_edge_kwargs(conv_type: str, edge_types, data: HeteroData, edge_mask_dict=None):
#     """edge_mask_dict: opcjonalne maski z DropEdge (bool, dlugosc = liczba
#     ORYGINALNYCH krawedzi danej relacji). Jesli podane, edge_attr/edge_weight
#     musi zostac odfiltrowany TA SAMA maska co edge_index - inaczej wiadomosci
#     dostana atrybuty nieodpowiadajacych im krawedzi (cichy, trudny do wykrycia blad)."""
#     edge_weight_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
#     edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
#     for edge_type in edge_types:
#         store = data[edge_type]
#         attr = getattr(store, "edge_attr", None)
#         if attr is None:
#             continue
#         if edge_mask_dict is not None and edge_type in edge_mask_dict:
#             attr = attr[edge_mask_dict[edge_type]]
#         if conv_type in _EDGE_WEIGHT_CONVS:
#             # edge_attr ma teraz wiele kolumn: [effect_size*effect_sign, activation_frequency,
#             # mechanistic_confidence, evidence_weight, edge_type_onehot...]. Konwolucje
#             # przyjmujace skalarny edge_weight (np. GraphConv) dostaja WYLACZNIE pierwsza
#             # kolumne (podpisana sila efektu). attr.view(-1) byloby bledem: splaszczyloby
#             # [E, F] do wektora dlugosci E*F zamiast E.
#             edge_weight_dict[edge_type] = attr[:, 0]
#         elif conv_type in _EDGE_ATTR_CONVS:
#             edge_attr_dict[edge_type] = attr
#     return {"edge_weight_dict": edge_weight_dict, "edge_attr_dict": edge_attr_dict}


# # ---------------------------------------------------------------------------
# # 0. Wspolna baza: input_proj, embedding tozsamosci wezla, stos HeteroConv
# # ---------------------------------------------------------------------------

# class _PatientDAGGNNBase(nn.Module):
#     """Wspolna logika dla wariantow Simple i Targeted, zeby nie duplikowac
#     (i nie rozjezdzac) tych samych poprawek w dwoch miejscach.

#     Zmiany wzgledem poprzedniej wersji:
#     - edge_dim jest jawnie przekazywany (np. z topology["edge_attr_dim"]),
#       zamiast domyslnego cfg.model.edge_dim=1, ktore po zmianie edge_attr
#       w build_patient_dag_heterodata.py (26 kolumn) powodowaloby blad
#       niezgodnosci ksztaltow w GATv2Conv/TransformerConv.
#     - HeteroConv nie zwraca typow wezlow bez zadnej wchodzacej krawedzi w danej
#       warstwie (np. patient_context, drug_exposure sa zrodlami w DAG-u).
#       Zamiast KeyError / cichej utraty tych typow od warstwy 2, ich reprezentacja
#       jest po prostu przenoszona bez zmian (nie ma nowej informacji do dodania).
#     - Embedding tozsamosci wezla (node_idx -> wektor), dodawany PO projekcji
#       wejsciowej. Rozwiazuje kolizje: wezly tego samego typu o identycznych
#       cechach statycznych (np. Serotonin_syndrome/Rhabdomyolysis/Lactic_acidosis
#       maja te sama [severity, observability, rarity, priority]) byly wczesniej
#       nierozroznialne dla modelu poza polozeniem w grafie.
#     - use_residual jest teraz jawnym przelacznikiem, nie przypadkowa roznica
#       miedzy klasami. Ma bezposrednie znaczenie teoretyczne: bez residual
#       operator per-warstwa to A (macierz sasiedztwa wazona), ktora na DAG-u
#       bez self-loopow jest nilpotentna (sygnal zanika, nie "wygladza sie").
#       Z residual operator to (I + A), co odpowiada klasycznemu oversmoothingowi.
#       Domyslnie True (zachowuje dawne zachowanie SimplePatientDAGNodeClassifier).
#     - self_transform (NOWE): jeden, kontrolowany self-sygnal per warstwa per
#       typ wezla, ZAMIAST pozwalac konwolucji dublowac go wewnetrznie R razy
#       (R = liczba relacji wchodzacych do danego typu). Dotyczy TYLKO
#       conv_type w {"sage","gat","transformer"} (patrz _SELF_TRANSFORM_CONVS):
#         * "sage"/"transformer": root_weight=False w _build_conv usuwa
#           wewnetrzny self-transform CALKOWICIE - self_transform tutaj jest
#           JEDYNYM self-sygnalem, use_residual ma pelna, czysta kontrole
#           (operator A vs I+A).
#         * "gat": GATv2Conv nie ma parametru root_weight - self-zaleznosc
#           wchodzi przez wektor zapytania w uwadze (query z x_dst), nie da
#           sie jej wylaczyc. self_transform jest tu DODATKOWYM, kontrolowanym
#           skladnikiem NA WIERZCHU tej wbudowanej zaleznosci - use_residual
#           steruje tylko czescia self-sygnalu, nie caloscia.
#         * "gcn": WYKLUCZONE z self_transform. GraphConv nie ma parametru
#           root_weight - ma WLASNY, nieusuwalny self-transform w kazdym
#           wywolaniu. Dolozenie zewnetrznego by to tylko zdublowalo (dokladnie
#           ten sam blad, ktory naprawiamy dla sage). use_residual jest wiec
#           dla "gcn" IGNOROWANY (ostrzezenie przy konstrukcji modelu).
#     """

#     def __init__(
#         self,
#         cfg,
#         metadata: Tuple[List[str], List[Tuple[str, str, str]]],
#         in_dims: Dict[str, int],
#         node_names_by_type: Dict[str, List[str]],
#         edge_dim: Optional[int] = None,
#     ):
#         super().__init__()
#         hidden_dim = int(cfg.model.hidden_dim)
#         num_layers = int(cfg.model.num_layers)
#         dropout = float(getattr(cfg.model, "dropout", 0.0))
#         aggr = str(getattr(cfg.model, "aggr", "sum"))
#         conv_type = str(getattr(cfg.model, "conv_type", "sage")).lower()

#         # edge_dim: preferuj jawny argument (np. topology["edge_attr_dim"]);
#         # cfg.model.edge_dim to tylko fallback dla wstecznej zgodnosci.
#         resolved_edge_dim = edge_dim if edge_dim is not None else int(getattr(cfg.model, "edge_dim", 1))

#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#         self.dropout = dropout
#         self.conv_type = conv_type
#         self.edge_dim = resolved_edge_dim
#         self.node_types = list(metadata[0])
#         self.edge_types = list(metadata[1])
#         self.use_residual = bool(getattr(cfg.model, "use_residual", True))
#         self.grad_clip = float(getattr(cfg.training, "grad_clip", 1.0))

#         if conv_type == "gcn" and self.use_residual:
#             warnings.warn(
#                 "conv_type='gcn' (GraphConv) nie ma parametru root_weight - "
#                 "ma wlasny, nieusuwalny self-transform w kazdej warstwie. "
#                 "use_residual=True jest dla tej architektury ignorowany "
#                 "(zewnetrzny self_transform nie jest dodawany, zeby nie "
#                 "zdublowac wbudowanego). Patrz docstring _PatientDAGGNNBase.",
#                 stacklevel=2,
#             )

#         self.input_proj = nn.ModuleDict({
#             node_type: nn.Linear(int(in_dims[node_type]), hidden_dim)
#             for node_type in self.node_types
#         })

#         # Embedding tozsamosci wezla: jeden wpis na kazdy konkretny wezel danego
#         # typu (np. embedding[42] = wektor dla wezla "nsaid" w typie drug_exposure).
#         # Dodawany do projekcji wejsciowej, wiec nie zmienia in_dims modelu.
#         self.node_embedding = nn.ModuleDict({
#             node_type: nn.Embedding(len(node_names_by_type[node_type]), hidden_dim)
#             for node_type in self.node_types
#         })

#         self.convs = nn.ModuleList()
#         for _ in range(num_layers):
#             conv_dict = {
#                 edge_type: _build_conv(conv_type, hidden_dim, cfg, edge_dim=resolved_edge_dim)
#                 for edge_type in self.edge_types
#             }
#             self.convs.append(HeteroConv(conv_dict, aggr=aggr))

#         # Jeden, kontrolowany self-sygnal per warstwa per typ wezla - patrz
#         # docstring klasy. None dla "gcn" (patrz _SELF_TRANSFORM_CONVS).
#         if conv_type in _SELF_TRANSFORM_CONVS:
#             self.self_transform: Optional[nn.ModuleList] = nn.ModuleList([
#                 nn.ModuleDict({nt: nn.Linear(hidden_dim, hidden_dim) for nt in self.node_types})
#                 for _ in range(num_layers)
#             ])
#         else:
#             self.self_transform = None

#         # --- Interwencje anty-oversmoothing (wszystkie domyslnie WYLACZONE,
#         # zeby dotychczasowe runy odtwarzaly sie bez zmian) ---
#         #
#         # PairNorm (Zhao & Akoglu 2020): po kazdej warstwie centruje i skaluje
#         # reprezentacje tak, by SUMARYCZNA odleglosc miedzy wezlami pozostala
#         # stala. Nie zapobiega mieszaniu informacji, ale uniemozliwia calkowity
#         # kolaps do jednego punktu - dokladnie to, co widac u Ciebie jako
#         # MAD -> 0 i feature_std -> 0.
#         self.pair_norm_scale = float(getattr(cfg.model, "pair_norm_scale", 0.0))
#         self.pair_norm = PairNorm(scale=self.pair_norm_scale) if self.pair_norm_scale > 0 else None
#         #
#         # DropEdge (Rong i in. 2020): losowe usuwanie czesci krawedzi w KAZDEJ
#         # warstwie, niezaleznie, tylko podczas treningu. Spowalnia mieszanie
#         # (kazda warstwa widzi rzadszy graf) i dziala jak regularyzacja.
#         self.drop_edge = float(getattr(cfg.model, "drop_edge", 0.0))
#         #
#         # JumpingKnowledge (Xu i in. 2018): zamiast brac wylacznie wyjscie
#         # OSTATNIEJ warstwy, laczy wyjscia WSZYSTKICH warstw. Wezel moze wiec
#         # "wybrac" reprezentacje z plytszej warstwy, jesli glebsza jest juz
#         # rozmyta - najbardziej bezposrednia obrona przed oversmoothingiem.
#         # Tryby: "cat" (konkatenacja, zmienia wymiar wejscia glowicy),
#         # "max" (elementwise max), "lstm" (uczona agregacja po warstwach).
#         jk_mode = getattr(cfg.model, "jk_mode", None)
#         self.jk_mode = str(jk_mode).lower() if jk_mode else None
#         if self.jk_mode in {"none", "null", ""}:
#             self.jk_mode = None
#         if self.jk_mode is not None:
#             self.jk = JumpingKnowledge(
#                 mode=self.jk_mode, channels=hidden_dim, num_layers=num_layers
#             )
#             head_in_dim = hidden_dim * num_layers if self.jk_mode == "cat" else hidden_dim
#         else:
#             self.jk = None
#             head_in_dim = hidden_dim

#         self.head = NodeClassificationHead(head_in_dim, dropout=dropout)

#     def encode(
#         self, data: HeteroData, return_layer_reprs: bool = False
#     ) -> Dict[str, torch.Tensor] | Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
#         """Zwraca reprezentacje koncowe per typ wezla.

#         Jesli return_layer_reprs=True, zwraca dodatkowo liste snapshotow
#         x_dict po kazdej warstwie (przydatne do liczenia metryk oversmoothingu
#         typu MAD / energia Dirichleta per warstwa, bez modyfikowania forward()).
#         """
#         x_dict = {
#             node_type: F.relu(
#                 self.input_proj[node_type](data[node_type].x)
#                 + self.node_embedding[node_type](data[node_type].node_idx)
#             )
#             for node_type in self.node_types
#         }

#         # Bez DropEdge topologia jest stala, wiec edge_kwargs liczymy RAZ.
#         # Z DropEdge musimy przeliczac je w kazdej warstwie (inne krawedzie =
#         # inne atrybuty), stad rozgalezienie ponizej.
#         use_drop_edge = self.drop_edge > 0.0 and self.training
#         static_edge_kwargs = (
#             None if use_drop_edge
#             else _build_edge_kwargs(self.conv_type, self.edge_types, data)
#         )

#         layer_reprs: List[Dict[str, torch.Tensor]] = [{
#             nt: x.detach() for nt, x in x_dict.items()
#         }] if return_layer_reprs else []

#         # Wyjscia kolejnych warstw per typ wezla - potrzebne dla JumpingKnowledge.
#         jk_inputs: Dict[str, List[torch.Tensor]] = {nt: [] for nt in self.node_types}

#         for layer_idx, conv in enumerate(self.convs):
#             if use_drop_edge:
#                 edge_index_dict, edge_mask_dict = {}, {}
#                 for edge_type, edge_index in data.edge_index_dict.items():
#                     kept_index, kept_mask = dropout_edge(
#                         edge_index, p=self.drop_edge, training=True
#                     )
#                     edge_index_dict[edge_type] = kept_index
#                     edge_mask_dict[edge_type] = kept_mask
#                 edge_kwargs = _build_edge_kwargs(
#                     self.conv_type, self.edge_types, data, edge_mask_dict
#                 )
#             else:
#                 edge_index_dict = data.edge_index_dict
#                 edge_kwargs = static_edge_kwargs

#             call_kwargs = {}
#             if edge_kwargs["edge_weight_dict"]:
#                 call_kwargs["edge_weight_dict"] = edge_kwargs["edge_weight_dict"]
#             if edge_kwargs["edge_attr_dict"]:
#                 call_kwargs["edge_attr_dict"] = edge_kwargs["edge_attr_dict"]
#             x_dict_new = conv(x_dict, edge_index_dict, **call_kwargs)

#             updated: Dict[str, torch.Tensor] = {}
#             for node_type, current in x_dict.items():
#                 new_val = x_dict_new.get(node_type)
#                 if new_val is None:
#                     # Ten typ wezla nie dostal zadnej wiadomosci w tej warstwie
#                     # (np. patient_context/drug_exposure jako zrodla DAG-a).
#                     # Nie ma nowej informacji do dodania - reprezentacja
#                     # przechodzi bez zmian, niezaleznie od use_residual/conv_type.
#                     updated[node_type] = current
#                     continue

#                 transformed = F.dropout(F.relu(new_val), p=self.dropout, training=self.training)

#                 if self.self_transform is None:
#                     # "gcn": GraphConv juz ma wlasny, nieusuwalny self-transform
#                     # wbudowany w new_val - nie dodajemy niczego wiecej.
#                     updated[node_type] = transformed
#                 elif self.use_residual:
#                     self_z = self.self_transform[layer_idx][node_type](current)
#                     updated[node_type] = self_z + transformed
#                 else:
#                     updated[node_type] = transformed

#             if self.pair_norm is not None:
#                 # batch: normalizacja MUSI byc per pacjent, nie po calym batchu -
#                 # inaczej "srednia reprezentacja" mieszalaby roznych pacjentow
#                 # i PairNorm przestalby znaczyc to, co ma znaczyc.
#                 for node_type in updated:
#                     batch_vec = getattr(data[node_type], "batch", None)
#                     updated[node_type] = self.pair_norm(updated[node_type], batch_vec)

#             x_dict = updated
#             for node_type in self.node_types:
#                 jk_inputs[node_type].append(x_dict[node_type])

#             if return_layer_reprs:
#                 layer_reprs.append({nt: x.detach() for nt, x in x_dict.items()})

#         if self.jk is not None:
#             x_dict = {nt: self.jk(jk_inputs[nt]) for nt in self.node_types}
#             if return_layer_reprs:
#                 # Ostatni snapshot to reprezentacja PO JK - to ona trafia do
#                 # glowicy, wiec metryki koncowe powinny dotyczyc wlasnie jej.
#                 layer_reprs.append({nt: x.detach() for nt, x in x_dict.items()})

#         if return_layer_reprs:
#             return x_dict, layer_reprs
#         return x_dict


# # ---------------------------------------------------------------------------
# # 1. Baseline: klasyfikuje WSZYSTKIE wezly typu target_node_type
# # ---------------------------------------------------------------------------

# class SimplePatientDAGNodeClassifier(_PatientDAGGNNBase):

#     def __init__(
#         self,
#         cfg,
#         metadata: Tuple[List[str], List[Tuple[str, str, str]]],
#         in_dims: Dict[str, int],
#         node_names_by_type: Dict[str, List[str]],
#         target_node_type: str = "clinical_endpoint",
#         edge_dim: Optional[int] = None,
#     ):
#         super().__init__(
#             cfg=cfg, metadata=metadata, in_dims=in_dims,
#             node_names_by_type=node_names_by_type, edge_dim=edge_dim,
#         )
#         self.target_node_type = target_node_type

#     def forward(self, data: HeteroData) -> torch.Tensor:
#         z_dict = self.encode(data)
#         target_z = z_dict[self.target_node_type]
#         return self.head(target_z)


# # ---------------------------------------------------------------------------
# # 2. Targeted: klasyfikuje TYLKO wybrana podliste endpointow
# # ---------------------------------------------------------------------------

# class TargetedPatientDAGNodeClassifier(_PatientDAGGNNBase):
#     """Baseline heterogeniczny GNN do node classification, z jawnym
#     ograniczeniem predykcji do wybranej podlisty wezlow-endpointow
#     (target_endpoint_names), w stalej kolejnosci.

#     use_hcr_wide (Wide&Deep, sekcja 9.2 dokumentu HCR): opcjonalna, PLYTKA
#     sciezka dodawana do logitu OBOK glebokiego GNN-a:

#         logit_e = GNN(graf)_e  +  waga_e * s_p,e

#     gdzie s_p,e to precomputed "wide" wynik HCR (binary-binary, leave-one-out
#     zabezpieczone przed leakage - patrz hcr_wide_features.py), dolaczony do
#     kazdego pacjenta jako data["clinical_endpoint"].hcr_wide. waga_e jest
#     JEDNYM, UCZONYM parametrem per endpoint, inicjalizowanym na 0 - model
#     startuje wiec IDENTYCZNIE jak bez HCR (czysty deep), a trening sam
#     decyduje, czy i jak bardzo zaufac sciezce wide dla kazdego endpointu
#     osobno. Bezpieczny domyslny stan: use_hcr_wide=False, brak zmiany
#     zachowania wzgledem wersji sprzed tej funkcji."""

#     def __init__(
#         self,
#         cfg,
#         metadata: Tuple[List[str], List[Tuple[str, str, str]]],
#         in_dims: Dict[str, int],
#         node_names_by_type: Dict[str, List[str]],
#         target_node_type: str = "clinical_endpoint",
#         target_endpoint_names: Optional[List[str]] = None,
#         edge_dim: Optional[int] = None,
#     ):
#         super().__init__(
#             cfg=cfg, metadata=metadata, in_dims=in_dims,
#             node_names_by_type=node_names_by_type, edge_dim=edge_dim,
#         )
#         self.target_node_type = target_node_type

#         all_endpoint_names = node_names_by_type[target_node_type]
#         requested = target_endpoint_names or getattr(cfg.data, "target_endpoints", None) or DEFAULT_TARGET_ENDPOINTS
#         requested = list(requested)

#         missing = [e for e in requested if e not in all_endpoint_names]
#         if missing:
#             raise ValueError(
#                 f"target_endpoint_names {missing} nie sa wezlami typu "
#                 f"'{target_node_type}' w DAG. Dostepne: {all_endpoint_names}"
#             )

#         self.target_endpoint_names = requested
#         name_to_local_idx = {name: i for i, name in enumerate(all_endpoint_names)}
#         self.register_buffer(
#             "target_local_idx",
#             torch.tensor([name_to_local_idx[e] for e in requested], dtype=torch.long),
#         )

#         self.use_hcr_wide = bool(getattr(cfg.model, "use_hcr_wide", False))
#         # hcr_wide_mode: "linear" (dotychczasowe: pojedyncza wyuczona waga per
#         # endpoint * s_p,e) albo "nonlinear" (HCRPairEncoder: maly wspoldzielony
#         # MLP koduje kazdego rodzica osobno, maskowana srednia po rodzicach,
#         # potem liniowy head -> skalar). Patrz uzasadnienie architektoniczne
#         # przy klasie HCRPairEncoder w gnn_common.py - plaska suma/konkatenacja
#         # przed nieliniowoscia byla najgorszym wariantem w analogicznej ablacji
#         # Task A (B2: gorzej niz brak HCR w ogole).
#         self.hcr_wide_mode = str(getattr(cfg.model, "hcr_wide_mode", "linear")).lower()
#         if self.use_hcr_wide and self.hcr_wide_mode not in {"linear", "nonlinear"}:
#             raise ValueError(
#                 f"cfg.model.hcr_wide_mode={self.hcr_wide_mode!r} nieznany. "
#                 "Dostepne: 'linear', 'nonlinear'."
#             )

#         self.hcr_wide_weight = None
#         self.hcr_pair_encoder = None
#         if self.use_hcr_wide and self.hcr_wide_mode == "linear":
#             # Inicjalizacja na zero: model startuje identycznie jak bez HCR,
#             # trening sam odkrywa, czy sciezka wide cokolwiek wnosi per endpoint.
#             self.hcr_wide_weight = nn.Parameter(torch.zeros(len(requested)))
#         elif self.use_hcr_wide and self.hcr_wide_mode == "nonlinear":
#             hcr_hidden_dim = int(getattr(cfg.model, "hcr_hidden_dim", 8))
#             hcr_embed_dim = int(getattr(cfg.model, "hcr_embed_dim", 4))
#             hcr_dropout = float(getattr(cfg.model, "hcr_dropout", 0.0))
#             self.hcr_pair_encoder = HCRPairEncoder(
#                 in_channels=N_PAIR_EVIDENCE_CHANNELS,
#                 hidden_dim=hcr_hidden_dim, embed_dim=hcr_embed_dim, dropout=hcr_dropout,
#             )
#             # out_head w HCRPairEncoder jest zwykla warstwa Linear (nie
#             # inicjalizowana na zero jak w wariancie liniowym) - PyTorch domyslnie
#             # daje jej mala, losowa wage (Kaiming/uniform), wiec wklad do logitu
#             # na starcie jest niewielki, ale NIE dokladnie zerowy jak w
#             # wariancie linear. To swiadoma roznica: MLP z zerowa inicjalizacja
#             # wszystkich warstw nie uczy sie w ogole (martwy gradient przez
#             # symetrie wag), wiec pelne "startuje identycznie jak bez HCR" nie
#             # jest tu osiagalne bez utraty trenowalnosci calej podsieci.

#     def _select_targets(self, full_tensor: torch.Tensor, batch_size: int, n_endpoint_nodes_per_graph: int) -> torch.Tensor:
#         offsets = torch.arange(batch_size, device=full_tensor.device) * n_endpoint_nodes_per_graph
#         idx = (offsets.unsqueeze(1) + self.target_local_idx.unsqueeze(0)).view(-1)
#         return full_tensor[idx]

#     def forward(self, data: HeteroData) -> torch.Tensor:
#         z_dict = self.encode(data)
#         target_z = z_dict[self.target_node_type]
#         logits_all = self.head(target_z)

#         batch_size = int(data[self.target_node_type].batch.max().item()) + 1 \
#             if hasattr(data[self.target_node_type], "batch") else 1
#         n_per_graph = target_z.size(0) // max(batch_size, 1)

#         if batch_size > 1:
#             deep_logits = self._select_targets(logits_all, batch_size, n_per_graph)
#         else:
#             deep_logits = logits_all[self.target_local_idx]

#         if not self.use_hcr_wide:
#             return deep_logits

#         if self.hcr_wide_mode == "linear":
#             # get_targeted_wide_scores uzywa DOKLADNIE tej samej logiki
#             # indeksowania (offsets + target_local_idx) co _select_targets
#             # powyzej, wiec wide jest juz w tym samym ksztalcie/kolejnosci co
#             # deep_logits - nie trzeba dodatkowego dopasowywania.
#             wide = get_targeted_wide_scores(data, self.target_local_idx, self.target_node_type)
#             weight = self.hcr_wide_weight.repeat(batch_size) if batch_size > 1 else self.hcr_wide_weight
#             return deep_logits + weight * wide

#         # hcr_wide_mode == "nonlinear": ta sama logika indeksowania
#         # (get_targeted_pair_evidence uzywa offsets+target_local_idx identycznie
#         # jak get_targeted_wide_scores/_select_targets), wiec features/mask sa
#         # juz w ksztalcie [batch*n_targets, max_parents, 9] - zgodnym z
#         # deep_logits [batch*n_targets] po HCRPairEncoder.forward().
#         features, mask = get_targeted_pair_evidence(data, self.target_local_idx, self.target_node_type)
#         hcr_contribution = self.hcr_pair_encoder(features, mask)
#         return deep_logits + hcr_contribution


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

from src.models.TaskB.gnn_common_20260806 import (
    DEFAULT_TARGET_ENDPOINTS,
    NodeClassificationHead,
    HCRPairEncoder,
    get_targeted_labels,
    get_targeted_wide_scores,
    get_targeted_pair_evidence,
    get_node_labels,
    get_node_mask,
)
from src.data.PreprocessingTaskB.hcr_wide_features_20260806 import N_PAIR_EVIDENCE_CHANNELS

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
        # root_weight=False: SAGEConv domyslnie dodaje WLASNY, wewnetrzny
        # self-transform (W_root @ x_dst) do kazdej wiadomosci. Owiniete w
        # HeteroConv per-relacja, to sie duplikuje R razy (R = liczba relacji
        # wchodzacych do danego typu wezla) - kazda relacja ma WLASNA macierz
        # W_root. root_weight=False usuwa to calkowicie; jedyny self-sygnal
        # pochodzi teraz z self_transform w _PatientDAGGNNBase (jeden,
        # kontrolowany, per warstwa per typ wezla - patrz encode()).
        return SAGEConv((hidden_dim, hidden_dim), hidden_dim, root_weight=False)
    if conv_type == "gcn":
        # GraphConv NIE MA parametru root_weight w PyG - self-transform
        # (lin_root) jest wbudowany bezwarunkowo, bez mozliwosci wylaczenia.
        # Duplikacja R-krotna (per relacja w HeteroConv) jest wiec dla tej
        # architektury NIEUSUWALNA bez pisania wlasnej warstwy od zera.
        # encode() NIE dodaje dla "gcn" zadnego dodatkowego self_transform
        # (dublowaloby to jeszcze bardziej) - use_residual jest dla "gcn"
        # ignorowany, z ostrzezeniem przy konstrukcji modelu.
        return GraphConv(hidden_dim, hidden_dim, aggr="add")
    if conv_type == "gat":
        # GATv2Conv (podobnie jak GAT) NIE MA parametru root_weight. Self-
        # zaleznosc wchodzi inna droga: wektor zapytania (query) w mechanizmie
        # uwagi jest liczony z x_dst, wiec cechy wezla docelowego ksztaltuja
        # WAGI uwagi dla sasiadow, nawet bez add_self_loops i bez osobnego
        # skladnika addytywnego. To NIE jest to samo zjawisko co w SAGEConv
        # (tam self byl DODAWANY wprost, tu tylko WPLYWA na wagi agregacji) -
        # nie da sie tego wylaczyc parametrem konstruktora. self_transform w
        # encode() dziala tu wiec jako DODATKOWY, kontrolowany self-sygnal
        # NA WIERZCHU tej wbudowanej, nieusuwalnej zaleznosci - use_residual
        # steruje tylko ta dodatkowa czescia, nie eliminuje zaleznosci
        # bazowej. Patrz docstring _PatientDAGGNNBase.
        heads = int(getattr(cfg.model, "heads", 4))
        return GATv2Conv(
            (hidden_dim, hidden_dim), hidden_dim // heads, heads=heads,
            add_self_loops=False, edge_dim=edge_dim,
        )
    if conv_type == "transformer":
        # root_weight=False: TransformerConv ma ten sam addytywny self-skip
        # co SAGEConv (mozliwy do wylaczenia), ALE dodatkowo, tak jak GAT,
        # liczy zapytanie w mechanizmie uwagi z x_dst - ta czesc zostaje
        # nieusuwalna niezaleznie od root_weight. root_weight=False usuwa
        # WIEC TYLKO addytywny skip, nie cala self-zaleznosc - analogiczne
        # zastrzezenie jak dla "gat" powyzej.
        heads = int(getattr(cfg.model, "heads", 4))
        return TransformerConv(
            (hidden_dim, hidden_dim), hidden_dim // heads, heads=heads,
            edge_dim=edge_dim, root_weight=False,
        )




def _build_edge_kwargs(conv_type: str, edge_types, data: HeteroData, edge_mask_dict=None):
    """edge_mask_dict: opcjonalne maski z DropEdge (bool, dlugosc = liczba
    ORYGINALNYCH krawedzi danej relacji). Jesli podane, edge_attr/edge_weight
    musi zostac odfiltrowany TA SAMA maska co edge_index - inaczej wiadomosci
    dostana atrybuty nieodpowiadajacych im krawedzi (cichy, trudny do wykrycia blad)."""
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
    """Wspolna logika dla wariantow Simple i Targeted, zeby nie duplikowac
    (i nie rozjezdzac) tych samych poprawek w dwoch miejscach.

    Zmiany wzgledem poprzedniej wersji:
    - edge_dim jest jawnie przekazywany (np. z topology["edge_attr_dim"]),
      zamiast domyslnego cfg.model.edge_dim=1, ktore po zmianie edge_attr
      w build_patient_dag_heterodata.py (26 kolumn) powodowaloby blad
      niezgodnosci ksztaltow w GATv2Conv/TransformerConv.
    - HeteroConv nie zwraca typow wezlow bez zadnej wchodzacej krawedzi w danej
      warstwie (np. patient_context, drug_exposure sa zrodlami w DAG-u).
      Zamiast KeyError / cichej utraty tych typow od warstwy 2, ich reprezentacja
      jest po prostu przenoszona bez zmian (nie ma nowej informacji do dodania).
    - Embedding tozsamosci wezla (node_idx -> wektor), dodawany PO projekcji
      wejsciowej. Rozwiazuje kolizje: wezly tego samego typu o identycznych
      cechach statycznych (np. Serotonin_syndrome/Rhabdomyolysis/Lactic_acidosis
      maja te sama [severity, observability, rarity, priority]) byly wczesniej
      nierozroznialne dla modelu poza polozeniem w grafie.
    - use_residual jest teraz jawnym przelacznikiem, nie przypadkowa roznica
      miedzy klasami. Ma bezposrednie znaczenie teoretyczne: bez residual
      operator per-warstwa to A (macierz sasiedztwa wazona), ktora na DAG-u
      bez self-loopow jest nilpotentna (sygnal zanika, nie "wygladza sie").
      Z residual operator to (I + A), co odpowiada klasycznemu oversmoothingowi.
      Domyslnie True (zachowuje dawne zachowanie SimplePatientDAGNodeClassifier).
    - self_transform (NOWE): jeden, kontrolowany self-sygnal per warstwa per
      typ wezla, ZAMIAST pozwalac konwolucji dublowac go wewnetrznie R razy
      (R = liczba relacji wchodzacych do danego typu). Dotyczy TYLKO
      conv_type w {"sage","gat","transformer"} (patrz _SELF_TRANSFORM_CONVS):
        * "sage"/"transformer": root_weight=False w _build_conv usuwa
          wewnetrzny self-transform CALKOWICIE - self_transform tutaj jest
          JEDYNYM self-sygnalem, use_residual ma pelna, czysta kontrole
          (operator A vs I+A).
        * "gat": GATv2Conv nie ma parametru root_weight - self-zaleznosc
          wchodzi przez wektor zapytania w uwadze (query z x_dst), nie da
          sie jej wylaczyc. self_transform jest tu DODATKOWYM, kontrolowanym
          skladnikiem NA WIERZCHU tej wbudowanej zaleznosci - use_residual
          steruje tylko czescia self-sygnalu, nie caloscia.
        * "gcn": WYKLUCZONE z self_transform. GraphConv nie ma parametru
          root_weight - ma WLASNY, nieusuwalny self-transform w kazdym
          wywolaniu. Dolozenie zewnetrznego by to tylko zdublowalo (dokladnie
          ten sam blad, ktory naprawiamy dla sage). use_residual jest wiec
          dla "gcn" IGNOROWANY (ostrzezenie przy konstrukcji modelu).
    """

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

        # edge_dim: preferuj jawny argument (np. topology["edge_attr_dim"]);
        # cfg.model.edge_dim to tylko fallback dla wstecznej zgodnosci.
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

        # Embedding tozsamosci wezla: jeden wpis na kazdy konkretny wezel danego
        # typu (np. embedding[42] = wektor dla wezla "nsaid" w typie drug_exposure).
        # Dodawany do projekcji wejsciowej, wiec nie zmienia in_dims modelu.
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

        # Jeden, kontrolowany self-sygnal per warstwa per typ wezla - patrz
        # docstring klasy. None dla "gcn" (patrz _SELF_TRANSFORM_CONVS).
        if conv_type in _SELF_TRANSFORM_CONVS:
            self.self_transform: Optional[nn.ModuleList] = nn.ModuleList([
                nn.ModuleDict({nt: nn.Linear(hidden_dim, hidden_dim) for nt in self.node_types})
                for _ in range(num_layers)
            ])
        else:
            self.self_transform = None

        # --- Interwencje anty-oversmoothing (wszystkie domyslnie WYLACZONE,
        # zeby dotychczasowe runy odtwarzaly sie bez zmian) ---
        #
        # PairNorm (Zhao & Akoglu 2020): po kazdej warstwie centruje i skaluje
        # reprezentacje tak, by SUMARYCZNA odleglosc miedzy wezlami pozostala
        # stala. Nie zapobiega mieszaniu informacji, ale uniemozliwia calkowity
        # kolaps do jednego punktu - dokladnie to, co widac u Ciebie jako
        # MAD -> 0 i feature_std -> 0.
        self.pair_norm_scale = float(getattr(cfg.model, "pair_norm_scale", 0.0))
        self.pair_norm = PairNorm(scale=self.pair_norm_scale) if self.pair_norm_scale > 0 else None
        #
        # DropEdge (Rong i in. 2020): losowe usuwanie czesci krawedzi w KAZDEJ
        # warstwie, niezaleznie, tylko podczas treningu. Spowalnia mieszanie
        # (kazda warstwa widzi rzadszy graf) i dziala jak regularyzacja.
        self.drop_edge = float(getattr(cfg.model, "drop_edge", 0.0))
        #
        # JumpingKnowledge (Xu i in. 2018): zamiast brac wylacznie wyjscie
        # OSTATNIEJ warstwy, laczy wyjscia WSZYSTKICH warstw. Wezel moze wiec
        # "wybrac" reprezentacje z plytszej warstwy, jesli glebsza jest juz
        # rozmyta - najbardziej bezposrednia obrona przed oversmoothingiem.
        # Tryby: "cat" (konkatenacja, zmienia wymiar wejscia glowicy),
        # "max" (elementwise max), "lstm" (uczona agregacja po warstwach).
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
        """Zwraca reprezentacje koncowe per typ wezla.

        Jesli return_layer_reprs=True, zwraca dodatkowo liste snapshotow
        x_dict po kazdej warstwie (przydatne do liczenia metryk oversmoothingu
        typu MAD / energia Dirichleta per warstwa, bez modyfikowania forward()).
        """
        x_dict = {
            node_type: F.relu(
                self.input_proj[node_type](data[node_type].x)
                + self.node_embedding[node_type](data[node_type].node_idx)
            )
            for node_type in self.node_types
        }

        # Bez DropEdge topologia jest stala, wiec edge_kwargs liczymy RAZ.
        # Z DropEdge musimy przeliczac je w kazdej warstwie (inne krawedzie =
        # inne atrybuty), stad rozgalezienie ponizej.
        use_drop_edge = self.drop_edge > 0.0 and self.training
        static_edge_kwargs = (
            None if use_drop_edge
            else _build_edge_kwargs(self.conv_type, self.edge_types, data)
        )

        layer_reprs: List[Dict[str, torch.Tensor]] = [{
            nt: x.detach() for nt, x in x_dict.items()
        }] if return_layer_reprs else []

        # Wyjscia kolejnych warstw per typ wezla - potrzebne dla JumpingKnowledge.
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
                    # Ten typ wezla nie dostal zadnej wiadomosci w tej warstwie
                    # (np. patient_context/drug_exposure jako zrodla DAG-a).
                    # Nie ma nowej informacji do dodania - reprezentacja
                    # przechodzi bez zmian, niezaleznie od use_residual/conv_type.
                    updated[node_type] = current
                    continue

                transformed = F.dropout(F.relu(new_val), p=self.dropout, training=self.training)

                if self.self_transform is None:
                    # "gcn": GraphConv juz ma wlasny, nieusuwalny self-transform
                    # wbudowany w new_val - nie dodajemy niczego wiecej.
                    updated[node_type] = transformed
                elif self.use_residual:
                    self_z = self.self_transform[layer_idx][node_type](current)
                    updated[node_type] = self_z + transformed
                else:
                    updated[node_type] = transformed

            if self.pair_norm is not None:
                # batch: normalizacja MUSI byc per pacjent, nie po calym batchu -
                # inaczej "srednia reprezentacja" mieszalaby roznych pacjentow
                # i PairNorm przestalby znaczyc to, co ma znaczyc.
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
    """Baseline heterogeniczny GNN do node classification, z jawnym
    ograniczeniem predykcji do wybranej podlisty wezlow-endpointow
    (target_endpoint_names), w stalej kolejnosci.

    use_hcr_wide (Wide&Deep, sekcja 9.2 dokumentu HCR): opcjonalna, PLYTKA
    sciezka dodawana do logitu OBOK glebokiego GNN-a:

        logit_e = GNN(graf)_e  +  waga_e * s_p,e

    gdzie s_p,e to precomputed "wide" wynik HCR (binary-binary, leave-one-out
    zabezpieczone przed leakage - patrz hcr_wide_features.py), dolaczony do
    kazdego pacjenta jako data["clinical_endpoint"].hcr_wide. waga_e jest
    JEDNYM, UCZONYM parametrem per endpoint, inicjalizowanym na 0 - model
    startuje wiec IDENTYCZNIE jak bez HCR (czysty deep), a trening sam
    decyduje, czy i jak bardzo zaufac sciezce wide dla kazdego endpointu
    osobno. Bezpieczny domyslny stan: use_hcr_wide=False, brak zmiany
    zachowania wzgledem wersji sprzed tej funkcji."""

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
        # hcr_wide_mode: "linear" (dotychczasowe: pojedyncza wyuczona waga per
        # endpoint * s_p,e) albo "nonlinear" (HCRPairEncoder: maly wspoldzielony
        # MLP koduje kazdego rodzica osobno, maskowana srednia po rodzicach,
        # potem liniowy head -> skalar). Patrz uzasadnienie architektoniczne
        # przy klasie HCRPairEncoder w gnn_common.py - plaska suma/konkatenacja
        # przed nieliniowoscia byla najgorszym wariantem w analogicznej ablacji
        # Task A (B2: gorzej niz brak HCR w ogole).
        self.hcr_wide_mode = str(getattr(cfg.model, "hcr_wide_mode", "linear")).lower()
        if self.use_hcr_wide and self.hcr_wide_mode not in {"linear", "nonlinear"}:
            raise ValueError(
                f"cfg.model.hcr_wide_mode={self.hcr_wide_mode!r} nieznany. "
                "Dostepne: 'linear', 'nonlinear'."
            )

        self.hcr_wide_weight = None
        self.hcr_pair_encoder = None
        if self.use_hcr_wide and self.hcr_wide_mode == "linear":
            # Inicjalizacja na zero: model startuje identycznie jak bez HCR,
            # trening sam odkrywa, czy sciezka wide cokolwiek wnosi per endpoint.
            self.hcr_wide_weight = nn.Parameter(torch.zeros(len(requested)))
        elif self.use_hcr_wide and self.hcr_wide_mode == "nonlinear":
            hcr_hidden_dim = int(getattr(cfg.model, "hcr_hidden_dim", 8))
            hcr_embed_dim = int(getattr(cfg.model, "hcr_embed_dim", 4))
            hcr_dropout = float(getattr(cfg.model, "hcr_dropout", 0.0))
            # MUSI odpowiadac funkcji uzytej przy budowie danych:
            #   9  -> compute_hcr_pair_evidence (podstawowa wersja)
            #   41 -> compute_hcr_pair_evidence_40d (wzbogacona, wg schematu Task A)
            # Niezgodnosc da czytelny blad ksztaltu przy pierwszym forward(),
            # nie ciche obciecie/wypelnienie danych.
            hcr_evidence_dim = int(getattr(cfg.model, "hcr_evidence_dim", N_PAIR_EVIDENCE_CHANNELS))
            self.hcr_pair_encoder = HCRPairEncoder(
                in_channels=hcr_evidence_dim,
                hidden_dim=hcr_hidden_dim, embed_dim=hcr_embed_dim, dropout=hcr_dropout,
            )
            # out_head w HCRPairEncoder jest zwykla warstwa Linear (nie
            # inicjalizowana na zero jak w wariancie liniowym) - PyTorch domyslnie
            # daje jej mala, losowa wage (Kaiming/uniform), wiec wklad do logitu
            # na starcie jest niewielki, ale NIE dokladnie zerowy jak w
            # wariancie linear. To swiadoma roznica: MLP z zerowa inicjalizacja
            # wszystkich warstw nie uczy sie w ogole (martwy gradient przez
            # symetrie wag), wiec pelne "startuje identycznie jak bez HCR" nie
            # jest tu osiagalne bez utraty trenowalnosci calej podsieci.

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

        if self.hcr_wide_mode == "linear":
            # get_targeted_wide_scores uzywa DOKLADNIE tej samej logiki
            # indeksowania (offsets + target_local_idx) co _select_targets
            # powyzej, wiec wide jest juz w tym samym ksztalcie/kolejnosci co
            # deep_logits - nie trzeba dodatkowego dopasowywania.
            wide = get_targeted_wide_scores(data, self.target_local_idx, self.target_node_type)
            weight = self.hcr_wide_weight.repeat(batch_size) if batch_size > 1 else self.hcr_wide_weight
            return deep_logits + weight * wide

        # hcr_wide_mode == "nonlinear": ta sama logika indeksowania
        # (get_targeted_pair_evidence uzywa offsets+target_local_idx identycznie
        # jak get_targeted_wide_scores/_select_targets), wiec features/mask sa
        # juz w ksztalcie [batch*n_targets, max_parents, 9] - zgodnym z
        # deep_logits [batch*n_targets] po HCRPairEncoder.forward().
        features, mask = get_targeted_pair_evidence(data, self.target_local_idx, self.target_node_type)
        hcr_contribution = self.hcr_pair_encoder(features, mask)
        return deep_logits + hcr_contribution

