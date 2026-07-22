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
    """Stacks `num_layers` HeteroConv(SAGEConv) blocks with per-node-type
    residual projections, generalizing the colleague's original fixed
    2-layer version to an arbitrary depth.

    Layer 0: in_channels -> hidden_channels
    Layer i>0: hidden_channels -> hidden_channels

    Each layer has its own residual Linear + HeteroConv, matching the
    original residual1/conv1, residual2/conv2 pattern but generalized into
    ModuleLists indexed by layer.
    """

    def __init__(self, data, in_channels: int, hidden_channels: int, num_layers: int = 2):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        relations = data.edge_types
        node_types = data.node_types
        self.num_layers = num_layers

        self.residuals = nn.ModuleList()
        self.convs = nn.ModuleList()

        for layer_idx in range(num_layers):
            layer_in = in_channels if layer_idx == 0 else hidden_channels
            self.residuals.append(
                nn.ModuleDict({node_type: nn.Linear(layer_in, hidden_channels) for node_type in node_types})
            )
            self.convs.append(
                HeteroConv(
                    {relation: SAGEConv((layer_in, layer_in), hidden_channels) for relation in relations},
                    aggr="sum",
                )
            )

    def forward(self, data) -> torch.Tensor:
        x_dict = data.x_dict
        for layer_idx in range(self.num_layers):
            messages = self.convs[layer_idx](x_dict, data.edge_index_dict)
            x_dict = {
                node_type: torch.relu(
                    self.residuals[layer_idx][node_type](x_dict[node_type]) + messages.get(node_type, 0.0)
                )
                for node_type in data.node_types
            }
        return flatten_embeddings(data, x_dict)


class ExplicitRGCN(nn.Module):
    """Stacks `num_layers` RGCNConv layers on the flattened (global-index)
    graph, generalizing the colleague's original fixed 2-layer version.

    Layer 0: in_channels -> hidden_channels
    Layer i>0: hidden_channels -> hidden_channels
    """

    def __init__(self, data, in_channels: int, hidden_channels: int, num_layers: int = 2, num_bases: int = 8):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.node_types = sorted(data.node_types)
        self.offsets, _ = global_layout(data)
        self.relations = list(data.edge_types)
        self.num_layers = num_layers

        bases = min(num_bases, len(self.relations))
        self.convs = nn.ModuleList()
        for layer_idx in range(num_layers):
            layer_in = in_channels if layer_idx == 0 else hidden_channels
            self.convs.append(
                RGCNConv(layer_in, hidden_channels, num_relations=len(self.relations), num_bases=bases)
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

        hidden = x
        for layer_idx, conv in enumerate(self.convs):
            hidden = conv(hidden, edge_index, edge_type)
            hidden = torch.relu(hidden)
        return hidden


class LinkDecoder(nn.Module):
    """Ported unchanged from the colleague's hetero_gnn_link_prediction_v2_2.py."""

    def __init__(self, hidden_channels: int, dropout: float = 0.20):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, embeddings: torch.Tensor, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        source_z = embeddings[source]
        target_z = embeddings[target]
        pair = torch.cat([source_z, target_z, source_z * target_z, torch.abs(source_z - target_z)], dim=1)
        return self.network(pair).squeeze(1)


class HeteroReconGNN(nn.Module):
    """Wraps the colleague's encoder + decoder behind our own model interface.

    cfg.model.name / cfg.model.conv_type selects "hetero_sage" or "rgcn".
    cfg.model.num_layers (default 2) controls encoder depth for both
    architectures -- sweep this to study oversmoothing.
    cfg.model.dropout (default 0.20) controls the decoder's dropout.
    """

    def __init__(self, cfg, data, in_channels: int, hidden_channels: int):
        super().__init__()
        arch = str(getattr(cfg.model, "conv_type", None) or cfg.model.name).lower()
        num_layers = int(getattr(cfg.model, "num_layers", 2))
        dropout = float(getattr(cfg.model, "dropout", 0.20))

        if arch in {"hetero_sage", "sage", "graphsage", "heterosage"}:
            self.encoder: nn.Module = NativeHeteroSAGE(
                data, in_channels, hidden_channels, num_layers=num_layers
            )
        elif arch in {"rgcn"}:
            num_bases = int(getattr(cfg.model, "num_bases", 8))
            self.encoder = ExplicitRGCN(
                data, in_channels, hidden_channels, num_layers=num_layers, num_bases=num_bases
            )
        else:
            raise ValueError(f"Unknown structural-recon architecture: {arch!r}")

        self.decoder = LinkDecoder(hidden_channels, dropout=dropout)
        self.num_layers = num_layers
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
