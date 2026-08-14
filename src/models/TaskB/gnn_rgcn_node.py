from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import RGCNConv

from models.TaskB.gnn_common import DEFAULT_TARGET_ENDPOINTS, NodeClassificationHead

from models.TaskB.gnn_common import (
    DEFAULT_TARGET_ENDPOINTS, NodeClassificationHead, HCRPairEncoder,
    get_targeted_wide_scores, get_targeted_pair_evidence,
    get_targeted_pair_evidence_named,
)
from models.TaskB.gnn_node import PARENT_NODE_TYPES
from data.PreprocessingTaskB.hcr_wide_features import N_PAIR_EVIDENCE_CHANNELS
from data.PreprocessingTaskB.hcr_triple_features import N_TRIPLE_EVIDENCE_CHANNELS

# Musi byc zsynchronizowane z build_patient_dag_heterodata.py: edge_attr ma
# layout [waga(1), activation_frequency, mechanistic_confidence,
# evidence_weight, edge_type_onehot(...)] - onehot zaczyna sie od kolumny 4.
# Uzywane do dekodowania edge_type z powrotem z one-hot bloku
# (patrz RGCNPatientDAGNodeClassifier._build_relation_map).
EDGE_ATTR_ONEHOT_START = 4


class RGCNPatientDAGNodeClassifier(nn.Module):

    def __init__(
        self,
        cfg,
        topology: dict,
        in_dims: Dict[str, int],
        target_node_type: str = "clinical_endpoint",
        target_endpoint_names: Optional[List[str]] = None,
    ):
        super().__init__()
        hidden_dim = int(cfg.model.hidden_dim)
        num_layers = int(cfg.model.num_layers)
        dropout = float(getattr(cfg.model, "dropout", 0.0))
        num_bases_requested = int(getattr(cfg.model, "num_bases", 8))

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_residual = bool(getattr(cfg.model, "use_residual", True))
        self.grad_clip = float(getattr(cfg.training, "grad_clip", 1.0))
        self.target_node_type = target_node_type
        # Zapamietane, bo tryb "typed_concat" nadpisuje glowice szersza
        # o wymiary blokow HCR.
        self.head_in_dim = hidden_dim
        self.head = NodeClassificationHead(hidden_dim, dropout=dropout)

        node_names_by_type = topology["node_names_by_type"]
        # Kolejnosc STALA (alfabetyczna) - splaszczanie (_flatten_batch) i
        # rozcinanie z powrotem (_split) musza uzywac tej samej kolejnosci.
        self.node_types = sorted(node_names_by_type)

        self.input_proj = nn.ModuleDict({
            nt: nn.Linear(int(in_dims[nt]), hidden_dim) for nt in self.node_types
        })
        self.node_embedding = nn.ModuleDict({
            nt: nn.Embedding(len(node_names_by_type[nt]), hidden_dim) for nt in self.node_types
        })

        self._build_relation_map(topology)

        num_bases = max(1, min(num_bases_requested, self.num_relations))
        self.convs = nn.ModuleList([
            RGCNConv(
                hidden_dim, hidden_dim, num_relations=self.num_relations,
                num_bases=num_bases, root_weight=False,
            )
            for _ in range(num_layers)
        ])
        self.self_transform = nn.ModuleList([
            nn.ModuleDict({nt: nn.Linear(hidden_dim, hidden_dim) for nt in self.node_types})
            for _ in range(num_layers)
        ])

        self.head = NodeClassificationHead(hidden_dim, dropout=dropout)

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
        # --- HCR (Wide&Deep) - identyczna logika jak w gnn_node.py -------
        self.use_hcr_wide = bool(getattr(cfg.model, "use_hcr_wide", False))
        self.hcr_wide_mode = str(getattr(cfg.model, "hcr_wide_mode", "linear")).lower()
        if self.use_hcr_wide and self.hcr_wide_mode not in {"linear", "nonlinear", "typed_concat", "triple"}:
            raise ValueError(f"cfg.model.hcr_wide_mode={self.hcr_wide_mode!r} nieznany.")

        self.hcr_wide_weight = None
        self.hcr_endpoint_weight = None
        self.hcr_type_gate = None
        self.hcr_pair_encoder = None
        self.hcr_type_encoders = None
        self.hcr_triple_encoder = None

        hcr_hidden_dim = int(getattr(cfg.model, "hcr_hidden_dim", 8))
        hcr_embed_dim = int(getattr(cfg.model, "hcr_embed_dim", 4))
        hcr_dropout = float(getattr(cfg.model, "hcr_dropout", 0.0))
        hcr_evidence_dim = int(getattr(cfg.model, "hcr_evidence_dim", N_PAIR_EVIDENCE_CHANNELS))

        if self.use_hcr_wide and self.hcr_wide_mode == "linear":
            self.hcr_wide_weight = nn.Parameter(torch.zeros(len(requested)))

        elif self.use_hcr_wide and self.hcr_wide_mode == "nonlinear":
            self.hcr_pair_encoder = HCRPairEncoder(
                hcr_evidence_dim, hcr_hidden_dim, hcr_embed_dim, hcr_dropout)
            self.hcr_endpoint_weight = nn.Parameter(torch.zeros(len(requested)))

        elif self.use_hcr_wide and self.hcr_wide_mode == "typed_concat":
            self.hcr_type_encoders = nn.ModuleDict({
                nt: HCRPairEncoder(hcr_evidence_dim, hcr_hidden_dim, hcr_embed_dim, hcr_dropout)
                for nt in PARENT_NODE_TYPES
            })
            self.hcr_type_gate = nn.Parameter(torch.ones(len(requested), len(PARENT_NODE_TYPES)))
            self.hcr_triple_encoder = HCRPairEncoder(
                N_TRIPLE_EVIDENCE_CHANNELS, hcr_hidden_dim, hcr_embed_dim, hcr_dropout)
            total_hcr_dim = hcr_embed_dim * (len(PARENT_NODE_TYPES) + 1)
            self.head = NodeClassificationHead(
                self.head_in_dim + total_hcr_dim, dropout=dropout)

        elif self.use_hcr_wide and self.hcr_wide_mode == "triple":
            self.hcr_pair_encoder = HCRPairEncoder(
                N_TRIPLE_EVIDENCE_CHANNELS, hcr_hidden_dim, hcr_embed_dim, hcr_dropout)

    # ------------------------------------------------------------------
    # Budowa mapy relacji: dekodowanie edge_type z one-hot bloku edge_attr,
    # RAZ, z danych topologii (te same krawedzie/wagi dla kazdego pacjenta).
    # ------------------------------------------------------------------

    def _build_relation_map(self, topology: dict) -> None:
        edge_attr_dict = topology["edge_attr_dict"]
        edge_index_dict_static = topology["edge_index_dict"]
        n_categories = len(topology["edge_type_categories"])
        onehot_start = EDGE_ATTR_ONEHOT_START

        relation_id_map: Dict[Tuple[str, str, int], int] = {}
        self._coarse_keys: List[Tuple[str, str, str]] = []
        self._n_edges_static: Dict[Tuple[str, str, str], int] = {}

        for coarse_key, attr in edge_attr_dict.items():
            src_type, _, dst_type = coarse_key
            onehot = attr[:, onehot_start: onehot_start + n_categories]
            row_sums = onehot.sum(dim=1)
            if not torch.allclose(row_sums, torch.ones_like(row_sums)):
                raise RuntimeError(
                    f"Relacja {coarse_key}: blok one-hot edge_type w edge_attr nie sumuje sie "
                    "do 1 dla kazdej krawedzi - niezgodnosc z build_patient_dag_heterodata.py "
                    "(sprawdz EDGE_ATTR_ONEHOT_START / kolejnosc kolumn edge_attr)."
                )
            category_idx = onehot.argmax(dim=1)

            fine_ids = torch.empty(category_idx.size(0), dtype=torch.long)
            for i in range(category_idx.size(0)):
                c = int(category_idx[i].item())
                key = (src_type, dst_type, c)
                if key not in relation_id_map:
                    relation_id_map[key] = len(relation_id_map)
                fine_ids[i] = relation_id_map[key]

            self.register_buffer(self._relation_buffer_name(coarse_key), fine_ids)
            self._coarse_keys.append(coarse_key)
            self._n_edges_static[coarse_key] = int(edge_index_dict_static[coarse_key].size(1))

        self.num_relations = len(relation_id_map)
        if self.num_relations == 0:
            raise RuntimeError("Nie znaleziono zadnych relacji do zbudowania R-GCN.")

    @staticmethod
    def _relation_buffer_name(coarse_key: Tuple[str, str, str]) -> str:
        src, _, dst = coarse_key
        return f"_finerel_{src}__{dst}"

    # ------------------------------------------------------------------
    # Splaszczanie/rozcinanie batcha
    # ------------------------------------------------------------------

    def _flatten_batch(
        self, data: HeteroData
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, int], Dict[str, int]]:
        starts: Dict[str, int] = {}
        sizes: Dict[str, int] = {}
        cursor = 0
        for nt in self.node_types:
            n_nodes = int(data[nt].x.size(0))
            starts[nt] = cursor
            sizes[nt] = n_nodes
            cursor += n_nodes

        x_parts = [
            F.relu(self.input_proj[nt](data[nt].x) + self.node_embedding[nt](data[nt].node_idx))
            for nt in self.node_types
        ]
        x_flat = torch.cat(x_parts, dim=0)

        edge_index_parts: List[torch.Tensor] = []
        edge_type_parts: List[torch.Tensor] = []
        for coarse_key in self._coarse_keys:
            edge_index_local = data.edge_index_dict.get(coarse_key)
            if edge_index_local is None or edge_index_local.numel() == 0:
                continue
            src_type, _, dst_type = coarse_key
            e_r_static = self._n_edges_static[coarse_key]
            e_r_batched = edge_index_local.size(1)
            if e_r_static == 0 or e_r_batched % e_r_static != 0:
                raise RuntimeError(
                    f"Relacja {coarse_key}: {e_r_batched} krawedzi po batchowaniu nie jest "
                    f"wielokrotnoscia {e_r_static} krawedzi z pojedynczego grafu (topology)."
                )
            repeat_factor = e_r_batched // e_r_static

            global_edge_index = edge_index_local.clone()
            global_edge_index[0] = global_edge_index[0] + starts[src_type]
            global_edge_index[1] = global_edge_index[1] + starts[dst_type]
            edge_index_parts.append(global_edge_index)

            fine_ids_static = getattr(self, self._relation_buffer_name(coarse_key))
            edge_type_parts.append(fine_ids_static.repeat(repeat_factor))

        edge_index_flat = torch.cat(edge_index_parts, dim=1)
        edge_type_flat = torch.cat(edge_type_parts, dim=0)
        return x_flat, edge_index_flat, edge_type_flat, starts, sizes

    def _split(
        self, x_flat: torch.Tensor, starts: Dict[str, int], sizes: Dict[str, int]
    ) -> Dict[str, torch.Tensor]:
        return {nt: x_flat[starts[nt]: starts[nt] + sizes[nt]] for nt in self.node_types}

    # ------------------------------------------------------------------
    # Encode / forward - ten sam interfejs zewnetrzny co architektury z gnn_node.py
    # ------------------------------------------------------------------

    def encode(
        self, data: HeteroData, return_layer_reprs: bool = False
    ) -> Dict[str, torch.Tensor] | Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        x_flat, edge_index_flat, edge_type_flat, starts, sizes = self._flatten_batch(data)

        layer_reprs: List[Dict[str, torch.Tensor]] = []
        if return_layer_reprs:
            layer_reprs.append({nt: t.detach() for nt, t in self._split(x_flat, starts, sizes).items()})

        current = x_flat
        for layer_idx, conv in enumerate(self.convs):
            conv_out = conv(current, edge_index_flat, edge_type_flat)
            conv_out = F.dropout(F.relu(conv_out), p=self.dropout, training=self.training)

            if self.use_residual:
                self_parts = [
                    self.self_transform[layer_idx][nt](current[starts[nt]: starts[nt] + sizes[nt]])
                    for nt in self.node_types
                ]
                current = torch.cat(self_parts, dim=0) + conv_out
            else:
                current = conv_out

            if return_layer_reprs:
                layer_reprs.append({nt: t.detach() for nt, t in self._split(current, starts, sizes).items()})

        z_dict = self._split(current, starts, sizes)
        if return_layer_reprs:
            return z_dict, layer_reprs
        return z_dict

    def _select_targets(self, full_tensor: torch.Tensor, batch_size: int, n_endpoint_nodes_per_graph: int) -> torch.Tensor:
        offsets = torch.arange(batch_size, device=full_tensor.device) * n_endpoint_nodes_per_graph
        idx = (offsets.unsqueeze(1) + self.target_local_idx.unsqueeze(0)).view(-1)
        return full_tensor[idx]

    def forward(self, data: HeteroData) -> torch.Tensor:
            z_dict = self.encode(data)
            target_z_all = z_dict[self.target_node_type]

            batch_size = int(data[self.target_node_type].batch.max().item()) + 1 \
                if hasattr(data[self.target_node_type], "batch") else 1
            n_per_graph = target_z_all.size(0) // max(batch_size, 1)

            if self.use_hcr_wide and self.hcr_wide_mode == "typed_concat":
                # Glowica ma szersze wejscie, wiec selekcja targetow MUSI nastapic
                # przed nia, nie po - inaczej reszta wezlow nie ma blokow HCR.
                if batch_size > 1:
                    target_z = self._select_targets(target_z_all, batch_size, n_per_graph)
                else:
                    target_z = target_z_all[self.target_local_idx]

                type_embeddings = []
                for i, node_type in enumerate(PARENT_NODE_TYPES):
                    features, mask = get_targeted_pair_evidence_named(
                        data, self.target_local_idx, self.target_node_type,
                        feature_attr=f"hcr_typed_{node_type}_features",
                        mask_attr=f"hcr_typed_{node_type}_mask",
                    )
                    embed = self.hcr_type_encoders[node_type].pooled_embedding(features, mask)
                    gate = self.hcr_type_gate[:, i]
                    gate = gate.repeat(batch_size) if batch_size > 1 else gate
                    type_embeddings.append(embed * gate.unsqueeze(-1))

                if self.hcr_triple_encoder is not None:
                    tf, tm = get_targeted_pair_evidence(
                        data, self.target_local_idx, self.target_node_type)
                    type_embeddings.append(self.hcr_triple_encoder.pooled_embedding(tf, tm))

                fused = torch.cat([target_z, torch.cat(type_embeddings, dim=-1)], dim=-1)
                return self.head(fused).squeeze(-1)

            logits_all = self.head(target_z_all)
            deep_logits = self._select_targets(logits_all, batch_size, n_per_graph) \
                if batch_size > 1 else logits_all[self.target_local_idx]

            if not self.use_hcr_wide:
                return deep_logits

            if self.hcr_wide_mode == "linear":
                wide = get_targeted_wide_scores(data, self.target_local_idx, self.target_node_type)
                weight = self.hcr_wide_weight.repeat(batch_size) if batch_size > 1 \
                    else self.hcr_wide_weight
                return deep_logits + weight * wide

            if self.hcr_wide_mode in ("nonlinear", "triple"):
                features, mask = get_targeted_pair_evidence(
                    data, self.target_local_idx, self.target_node_type)
                hcr_contribution = self.hcr_pair_encoder(features, mask)
                if self.hcr_endpoint_weight is not None:
                    weight = self.hcr_endpoint_weight.repeat(batch_size) if batch_size > 1 \
                        else self.hcr_endpoint_weight
                    hcr_contribution = weight * hcr_contribution
                return deep_logits + hcr_contribution

            raise ValueError(f"Nieznany hcr_wide_mode: {self.hcr_wide_mode!r}")
