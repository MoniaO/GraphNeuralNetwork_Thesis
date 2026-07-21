"""
Model wrapper for Task A (structural link prediction / graph reconstruction),
adapting the colleague's NativeHeteroSAGE / ExplicitRGCN / LinkDecoder code
so it exposes the same interface as our own models (SimpleHeteroGNN, LinearHeteroLP):

    model.encode(data)  -> dict[node_type] -> Tensor   (for oversmoothing metrics)
    model.forward(data) -> Tensor of logits, one per (link_source_idx, link_target_idx) pair

The colleague's graph has no `patient` node type and no fixed
("patient", "has_adr", "variable") edge type. Instead of forcing that
schema, candidate pairs are stored as *global* flat-index tensors directly
on the HeteroData object:

    data.link_source_idx : LongTensor [num_candidates]
    data.link_target_idx : LongTensor [num_candidates]
    data.edge_label       : FloatTensor [num_candidates]

This mirrors how the colleague's own loader already attaches graph-level
scalars (data.graph_version, data.scenario) directly on HeteroData.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import HeteroConv, RGCNConv, SAGEConv


def global_layout(data) -> tuple[dict[str, int], dict[str, int]]:
    """Same helper as in hetero_gnn_link_prediction_v2_2.py: assigns every
    node, across all node types, a single flat index (types sorted for a
    stable, reproducible layout)."""
    offsets: dict[str, int] = {}
    lookup: dict[str, int] = {}
    offset = 0
    for node_type in sorted(data.node_types):
        offsets[node_type] = offset
        for local_index, name in enumerate(data[node_type].node_name):
            lookup[name] = offset + local_index
        offset += data[node_type].num_nodes
    return offsets, lookup


def flatten_embeddings(data, embedding_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([embedding_dict[node_type] for node_type in sorted(data.node_types)], dim=0)


class NativeHeteroSAGE(nn.Module):
    """Ported unchanged from the colleague's hetero_gnn_link_prediction_v2_2.py."""

    def __init__(self, data, in_channels: int, hidden_channels: int):
        super().__init__()
        relations = data.edge_types
        self.residual1 = nn.ModuleDict(
            {node_type: nn.Linear(in_channels, hidden_channels) for node_type in data.node_types}
        )
        self.residual2 = nn.ModuleDict(
            {node_type: nn.Linear(hidden_channels, hidden_channels) for node_type in data.node_types}
        )
        self.conv1 = HeteroConv(
            {relation: SAGEConv((in_channels, in_channels), hidden_channels) for relation in relations},
            aggr="sum",
        )
        self.conv2 = HeteroConv(
            {relation: SAGEConv((hidden_channels, hidden_channels), hidden_channels) for relation in relations},
            aggr="sum",
        )

    def forward(self, data) -> torch.Tensor:
        messages1 = self.conv1(data.x_dict, data.edge_index_dict)
        hidden = {
            node_type: torch.relu(self.residual1[node_type](data[node_type].x) + messages1.get(node_type, 0.0))
            for node_type in data.node_types
        }
        messages2 = self.conv2(hidden, data.edge_index_dict)
        output = {
            node_type: torch.relu(self.residual2[node_type](hidden[node_type]) + messages2.get(node_type, 0.0))
            for node_type in data.node_types
        }
        return flatten_embeddings(data, output)


class ExplicitRGCN(nn.Module):
    """Ported unchanged from the colleague's hetero_gnn_link_prediction_v2_2.py."""

    def __init__(self, data, in_channels: int, hidden_channels: int):
        super().__init__()
        self.node_types = sorted(data.node_types)
        self.offsets, _ = global_layout(data)
        self.relations = list(data.edge_types)
        self.conv1 = RGCNConv(
            in_channels, hidden_channels,
            num_relations=len(self.relations), num_bases=min(8, len(self.relations)),
        )
        self.conv2 = RGCNConv(
            hidden_channels, hidden_channels,
            num_relations=len(self.relations), num_bases=min(8, len(self.relations)),
        )

    def forward(self, data) -> torch.Tensor:
        x = torch.cat([data[node_type].x for node_type in self.node_types], dim=0)
        edge_parts, relation_parts = [], []
        for relation_id, key in enumerate(self.relations):
            edge_index = data[key].edge_index.clone()
            edge_index[0] += self.offsets[key[0]]
            edge_index[1] += self.offsets[key[2]]
            edge_parts.append(edge_index)
            relation_parts.append(torch.full((edge_index.shape[1],), relation_id, dtype=torch.long))
        edge_index = torch.cat(edge_parts, dim=1)
        edge_type = torch.cat(relation_parts)
        hidden = torch.relu(self.conv1(x, edge_index, edge_type))
        return torch.relu(self.conv2(hidden, edge_index, edge_type))


class LinkDecoder(nn.Module):
    """Ported unchanged from the colleague's hetero_gnn_link_prediction_v2_2.py."""

    def __init__(self, hidden_channels: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, embeddings: torch.Tensor, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        source_z = embeddings[source]
        target_z = embeddings[target]
        pair = torch.cat([source_z, target_z, source_z * target_z, torch.abs(source_z - target_z)], dim=1)
        return self.network(pair).squeeze(1)


class HeteroReconGNN(nn.Module):
    """Wraps the colleague's encoder + decoder behind our own model interface.

    cfg.model.name / cfg.model.conv_type selects "hetero_sage" or "rgcn",
    exactly like the colleague's --architecture flag.
    """

    def __init__(self, cfg, data, in_channels: int, hidden_channels: int):
        super().__init__()
        arch = str(getattr(cfg.model, "conv_type", None) or cfg.model.name).lower()
        if arch in {"hetero_sage", "sage", "graphsage", "heterosage"}:
            self.encoder: nn.Module = NativeHeteroSAGE(data, in_channels, hidden_channels)
        elif arch in {"rgcn"}:
            self.encoder = ExplicitRGCN(data, in_channels, hidden_channels)
        else:
            raise ValueError(f"Unknown structural-recon architecture: {arch!r}")
        self.decoder = LinkDecoder(hidden_channels)
        self._node_types_sorted = sorted(data.node_types)
        self._sizes = {nt: data[nt].num_nodes for nt in data.node_types}
        grad_clip = float(getattr(cfg.training, "grad_clip", 0.0) or 0.0)
        self.grad_clip = grad_clip if grad_clip > 0 else None

    def encode(self, data) -> dict[str, torch.Tensor]:
        """Return per-node-type embeddings, matching the dict format expected
        by compute_oversmoothing_metrics / SynEvaluator."""
        flat = self.encoder(data)
        z_dict, start = {}, 0
        for node_type in self._node_types_sorted:
            n = self._sizes[node_type]
            z_dict[node_type] = flat[start : start + n]
            start += n
        return z_dict

    def forward(self, data) -> torch.Tensor:
        flat = self.encoder(data)
        return self.decoder(flat, data.link_source_idx, data.link_target_idx)
