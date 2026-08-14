from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, GraphConv, GATv2Conv, TransformerConv

from src.models.TaskB.gnn_common_20260806 import (
    DEFAULT_TARGET_ENDPOINTS,
    NodeClassificationHead,
    get_targeted_labels,
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
        return SAGEConv((hidden_dim, hidden_dim), hidden_dim, root_weight=False)
    if conv_type == "gcn":
        # GraphConv NIE MA parametru root_weight w PyG 
        return GraphConv(hidden_dim, hidden_dim, aggr="add")
    if conv_type == "gat":
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


def _build_edge_kwargs(conv_type: str, edge_types, data: HeteroData):
    edge_weight_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
    edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
    for edge_type in edge_types:
        store = data[edge_type]
        attr = getattr(store, "edge_attr", None)
        if attr is None:
            continue
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

        self.head = NodeClassificationHead(hidden_dim, dropout=dropout)

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
        edge_kwargs = _build_edge_kwargs(self.conv_type, self.edge_types, data)

        layer_reprs: List[Dict[str, torch.Tensor]] = [{
            nt: x.detach() for nt, x in x_dict.items()
        }] if return_layer_reprs else []

        for layer_idx, conv in enumerate(self.convs):
            call_kwargs = {}
            if edge_kwargs["edge_weight_dict"]:
                call_kwargs["edge_weight_dict"] = edge_kwargs["edge_weight_dict"]
            if edge_kwargs["edge_attr_dict"]:
                call_kwargs["edge_attr_dict"] = edge_kwargs["edge_attr_dict"]
            x_dict_new = conv(x_dict, data.edge_index_dict, **call_kwargs)

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
            x_dict = updated

            if return_layer_reprs:
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
    (target_endpoint_names), w stalej kolejnosci."""

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