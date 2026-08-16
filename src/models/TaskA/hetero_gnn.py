"""Task A structural link-prediction models.

The module contains:

- NativeHeteroSAGE:
    relation-specific GraphSAGE message passing on HeteroData;

- ExplicitRGCN:
    R-GCN operating on a flattened graph with explicit relation IDs;

- LinkDecoder:
    scores directed source-target candidate pairs;

- HeteroReconGNN:
    common Task A interface selecting the encoder from Hydra config.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch_geometric.nn import HeteroConv, RGCNConv, SAGEConv

from models.TaskA.factory import build_taskA_encoder
from utils.task_a_layout import (
    flatten_embeddings,
    global_layout,
    layout_signature,
)


MATCHED_ENCODER_NAMES = {
    "hetero_sage_matched",
    "sage_matched",
    "hetero_gatv2",
    "gatv2",
    "hgt",
    "rgcn_matched",
    "hetero_rgcn_matched",
}


class FlatDictEncoderAdapter(nn.Module):
    """Adapt a z_dict encoder to the flat embedding interface used by LinkDecoder."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, data: Any) -> torch.Tensor:
        embedding_dict = self.encoder(data)
        return flatten_embeddings(data=data, embedding_dict=embedding_dict)

    def encode_dict(self, data: Any) -> dict[str, torch.Tensor]:
        return self.encoder(data)


def _validate_feature_dimensions(
    data: Any,
    expected_in_channels: int,
) -> None:
    """Check that every node type has the expected feature dimension."""

    missing_features = [
        node_type
        for node_type in data.node_types
        if not hasattr(data[node_type], "x")
        or data[node_type].x is None
    ]

    if missing_features:
        raise AttributeError(
            "The following node types do not contain feature tensors `x`: "
            f"{sorted(missing_features)}"
        )

    feature_dimensions = {
        node_type: int(
            data[node_type].x.size(-1)
        )
        for node_type in data.node_types
    }

    incorrect_dimensions = {
        node_type: dimension
        for node_type, dimension in feature_dimensions.items()
        if dimension != expected_in_channels
    }

    if incorrect_dimensions:
        raise ValueError(
            "All node types must use the same input feature dimension. "
            f"Expected {expected_in_channels}, received: "
            f"{incorrect_dimensions}"
        )


class NativeHeteroSAGE(nn.Module):
    """Relation-specific heterogeneous GraphSAGE encoder.

    Every layer calculates:

        one residual transformation of the target node
        +
        relation-specific neighbor messages
        +
        ReLU

    SAGEConv uses ``root_weight=False`` because the node's own representation
    is already handled exactly once by the explicit residual projection.
    """

    def __init__(
        self,
        data: Any,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError(
                "num_layers must be at least 1, "
                f"received {num_layers}."
            )

        if in_channels < 1:
            raise ValueError(
                "in_channels must be positive, "
                f"received {in_channels}."
            )

        if hidden_channels < 1:
            raise ValueError(
                "hidden_channels must be positive, "
                f"received {hidden_channels}."
            )

        _validate_feature_dimensions(
            data=data,
            expected_in_channels=in_channels,
        )

        self.node_types = sorted(
            data.node_types
        )

        self.relations = sorted(
            data.edge_types
        )

        if not self.relations:
            raise ValueError(
                "NativeHeteroSAGE requires at least one edge relation."
            )

        self.num_layers = num_layers

        self.residuals = nn.ModuleList()
        self.convs = nn.ModuleList()

        for layer_index in range(num_layers):
            layer_input_channels = (
                in_channels
                if layer_index == 0
                else hidden_channels
            )

            self.residuals.append(
                nn.ModuleDict(
                    {
                        node_type: nn.Linear(
                            layer_input_channels,
                            hidden_channels,
                        )
                        for node_type in self.node_types
                    }
                )
            )

            self.convs.append(
                HeteroConv(
                    {
                        relation: SAGEConv(
                            (
                                layer_input_channels,
                                layer_input_channels,
                            ),
                            hidden_channels,
                            root_weight=False,
                        )
                        for relation in self.relations
                    },
                    aggr="sum",
                )
            )

    def forward(
        self,
        data: Any,
    ) -> torch.Tensor:
        """Encode all graph nodes and return one flat embedding matrix."""

        x_dict = data.x_dict

        for layer_index in range(self.num_layers):
            relation_messages = self.convs[
                layer_index
            ](
                x_dict,
                data.edge_index_dict,
            )

            updated_x_dict: dict[
                str,
                torch.Tensor,
            ] = {}

            for node_type in self.node_types:
                residual = self.residuals[
                    layer_index
                ][node_type](
                    x_dict[node_type]
                )

                message = relation_messages.get(
                    node_type
                )

                if message is None:
                    combined = residual
                else:
                    combined = (
                        residual
                        + message
                    )

                updated_x_dict[node_type] = torch.relu(
                    combined
                )

            x_dict = updated_x_dict

        return flatten_embeddings(
            data=data,
            embedding_dict=x_dict,
        )


class ExplicitRGCN(nn.Module):
    """R-GCN encoder using global node indices and explicit relation IDs."""

    def __init__(
        self,
        data: Any,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 2,
        num_bases: int = 8,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError(
                "num_layers must be at least 1, "
                f"received {num_layers}."
            )

        if num_bases < 1:
            raise ValueError(
                "num_bases must be at least 1, "
                f"received {num_bases}."
            )

        if in_channels < 1:
            raise ValueError(
                "in_channels must be positive, "
                f"received {in_channels}."
            )

        if hidden_channels < 1:
            raise ValueError(
                "hidden_channels must be positive, "
                f"received {hidden_channels}."
            )

        _validate_feature_dimensions(
            data=data,
            expected_in_channels=in_channels,
        )

        self.node_types = sorted(
            data.node_types
        )

        self.offsets, _ = global_layout(
            data
        )

        self.relations = sorted(
            data.edge_types
        )

        if not self.relations:
            raise ValueError(
                "ExplicitRGCN requires at least one edge relation."
            )

        self.num_layers = num_layers

        number_of_relations = len(
            self.relations
        )

        number_of_bases = min(
            num_bases,
            number_of_relations,
        )

        self.convs = nn.ModuleList()

        for layer_index in range(num_layers):
            layer_input_channels = (
                in_channels
                if layer_index == 0
                else hidden_channels
            )

            self.convs.append(
                RGCNConv(
                    in_channels=layer_input_channels,
                    out_channels=hidden_channels,
                    num_relations=number_of_relations,
                    num_bases=number_of_bases,
                )
            )

    def _flatten_graph(
        self,
        data: Any,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert heterogeneous edge indices to global node indices."""

        edge_parts: list[torch.Tensor] = []
        relation_parts: list[torch.Tensor] = []

        for relation_id, relation in enumerate(
            self.relations
        ):
            edge_index = (
                data[relation]
                .edge_index
                .to(
                    device=device,
                    dtype=torch.long,
                )
                .clone()
            )

            if edge_index.numel() == 0:
                continue

            source_type, _, target_type = relation

            edge_index[0] += self.offsets[
                source_type
            ]

            edge_index[1] += self.offsets[
                target_type
            ]

            edge_parts.append(
                edge_index
            )

            relation_parts.append(
                torch.full(
                    (
                        edge_index.size(1),
                    ),
                    relation_id,
                    dtype=torch.long,
                    device=device,
                )
            )

        if not edge_parts:
            raise ValueError(
                "The flattened R-GCN graph contains no edges."
            )

        flat_edge_index = torch.cat(
            edge_parts,
            dim=1,
        )

        flat_edge_type = torch.cat(
            relation_parts,
            dim=0,
        )

        return (
            flat_edge_index,
            flat_edge_type,
        )

    def forward(
        self,
        data: Any,
    ) -> torch.Tensor:
        """Encode all graph nodes in one flattened relational graph."""

        x = torch.cat(
            [
                data[node_type].x
                for node_type in self.node_types
            ],
            dim=0,
        )

        edge_index, edge_type = self._flatten_graph(
            data=data,
            device=x.device,
        )

        hidden = x

        for convolution in self.convs:
            hidden = convolution(
                hidden,
                edge_index,
                edge_type,
            )

            hidden = torch.relu(
                hidden
            )

        return hidden


class LinkDecoder(nn.Module):
    """Return one link-existence logit for every candidate pair."""

    def __init__(
        self,
        hidden_channels: int,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()

        if hidden_channels < 1:
            raise ValueError(
                "hidden_channels must be positive, "
                f"received {hidden_channels}."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in the interval [0, 1), "
                f"received {dropout}."
            )

        self.network = nn.Sequential(
            nn.Linear(
                4 * hidden_channels,
                hidden_channels,
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                hidden_channels,
                1,
            ),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits in exactly the same order as candidate pairs."""

        source = (
            source
            .to(
                device=embeddings.device,
                dtype=torch.long,
            )
            .view(-1)
        )

        target = (
            target
            .to(
                device=embeddings.device,
                dtype=torch.long,
            )
            .view(-1)
        )

        if source.numel() != target.numel():
            raise ValueError(
                "Candidate source and target tensors must have equal length. "
                f"Received {source.numel()} and {target.numel()}."
            )

        if source.numel() == 0:
            return embeddings.new_empty(
                (0,)
            )

        number_of_nodes = int(
            embeddings.size(0)
        )

        minimum_index = min(
            int(source.min().item()),
            int(target.min().item()),
        )

        maximum_index = max(
            int(source.max().item()),
            int(target.max().item()),
        )

        if minimum_index < 0:
            raise IndexError(
                f"Candidate index cannot be negative: {minimum_index}."
            )

        if maximum_index >= number_of_nodes:
            raise IndexError(
                "Candidate index exceeds embedding matrix size. "
                f"Maximum index: {maximum_index}; "
                f"number of embeddings: {number_of_nodes}."
            )

        source_embeddings = embeddings[
            source
        ]

        target_embeddings = embeddings[
            target
        ]

        pair_representation = torch.cat(
            [
                source_embeddings,
                target_embeddings,
                source_embeddings * target_embeddings,
                torch.abs(
                    source_embeddings
                    - target_embeddings
                ),
            ],
            dim=1,
        )

        logits = self.network(
            pair_representation
        )

        return logits.squeeze(
            dim=1
        )


class HCRLinkDecoder(nn.Module):
    """Link decoder with concatenated leakage-safe HCR pair features."""

    def __init__(
        self,
        hidden_channels: int,
        hcr_dim: int,
        dropout: float = 0.20,
        *,
        hidden_dims: list[int] | None = None,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        if hidden_channels < 1:
            raise ValueError(
                f"hidden_channels must be positive, received {hidden_channels}."
            )
        if hcr_dim < 1:
            raise ValueError(f"hcr_dim must be positive, received {hcr_dim}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                f"dropout must be in the interval [0, 1), received {dropout}."
            )

        from models.TaskA.mlp_builder import build_mlp

        self.hcr_dim = int(hcr_dim)
        self.hcr_norm = nn.LayerNorm(self.hcr_dim)
        pair_dim = 4 * hidden_channels + self.hcr_dim
        dims = list(hidden_dims) if hidden_dims is not None else [hidden_channels]
        self.network = build_mlp(
            input_dim=pair_dim,
            hidden_dims=dims,
            activation=activation,
            dropout=dropout,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        hcr_features: torch.Tensor,
    ) -> torch.Tensor:
        source = source.to(device=embeddings.device, dtype=torch.long).view(-1)
        target = target.to(device=embeddings.device, dtype=torch.long).view(-1)

        if source.numel() != target.numel():
            raise ValueError(
                "Candidate source and target tensors must have equal length. "
                f"Received {source.numel()} and {target.numel()}."
            )
        if source.numel() == 0:
            return embeddings.new_empty((0,))

        if hcr_features is None:
            raise ValueError("HCRLinkDecoder requires hcr_features.")
        if hcr_features.ndim != 2:
            raise ValueError(
                "hcr_features must have shape [num_candidates, hcr_dim]."
            )
        if int(hcr_features.size(0)) != int(source.numel()):
            raise ValueError(
                "hcr_features rows must match candidate count. "
                f"Received {hcr_features.size(0)} vs {source.numel()}."
            )
        if int(hcr_features.size(1)) != self.hcr_dim:
            raise ValueError(
                f"hcr_features dim {hcr_features.size(1)} != hcr_dim {self.hcr_dim}."
            )

        number_of_nodes = int(embeddings.size(0))
        minimum_index = min(int(source.min().item()), int(target.min().item()))
        maximum_index = max(int(source.max().item()), int(target.max().item()))
        if minimum_index < 0:
            raise IndexError(f"Candidate index cannot be negative: {minimum_index}.")
        if maximum_index >= number_of_nodes:
            raise IndexError(
                "Candidate index exceeds embedding matrix size. "
                f"Maximum index: {maximum_index}; "
                f"number of embeddings: {number_of_nodes}."
            )

        source_embeddings = embeddings[source]
        target_embeddings = embeddings[target]
        hcr = self.hcr_norm(
            hcr_features.to(device=embeddings.device, dtype=embeddings.dtype)
        )

        pair_representation = torch.cat(
            [
                source_embeddings,
                target_embeddings,
                source_embeddings * target_embeddings,
                torch.abs(source_embeddings - target_embeddings),
                hcr,
            ],
            dim=1,
        )
        return self.network(pair_representation).squeeze(dim=1)


class HeteroReconGNN(nn.Module):
    """Task A binary link-prediction encoder-decoder model."""

    def __init__(
        self,
        cfg: Any,
        data: Any,
        in_channels: int,
        hidden_channels: int,
    ) -> None:
        super().__init__()

        configured_architecture = getattr(
            cfg.model,
            "conv_type",
            None,
        )

        if configured_architecture is None:
            configured_architecture = getattr(
                cfg.model,
                "name",
                None,
            )

        if configured_architecture is None:
            raise ValueError(
                "Set cfg.model.conv_type or cfg.model.name."
            )

        architecture = str(
            configured_architecture
        ).strip().lower()

        # Nested model.hgt.* overrides (Wave 5D L2 architecture audit).
        hgt_cfg = getattr(cfg.model, "hgt", None)
        if hgt_cfg is not None:
            if getattr(hgt_cfg, "hidden_dim", None) is not None:
                hidden_channels = int(hgt_cfg.hidden_dim)
            if getattr(hgt_cfg, "num_layers", None) is not None:
                # Keep cfg.model.num_layers in sync for factory fallbacks.
                try:
                    from omegaconf import OmegaConf

                    OmegaConf.set_struct(cfg.model, False)
                    cfg.model.num_layers = int(hgt_cfg.num_layers)
                    cfg.model.hidden_dim = int(hidden_channels)
                    cfg.model.hidden_channels = int(hidden_channels)
                    if getattr(hgt_cfg, "heads", None) is not None:
                        cfg.model.heads = int(hgt_cfg.heads)
                    if getattr(hgt_cfg, "dropout", None) is not None:
                        cfg.model.dropout = float(hgt_cfg.dropout)
                    if getattr(hgt_cfg, "residual", None) is not None:
                        cfg.model.residual = bool(hgt_cfg.residual)
                except Exception:
                    pass

        number_of_layers = int(
            getattr(
                cfg.model,
                "num_layers",
                2,
            )
        )

        dropout = float(
            getattr(
                cfg.model,
                "dropout",
                0.20,
            )
        )

        if architecture in {
            "hetero_sage",
            "sage",
            "graphsage",
            "heterosage",
        }:
            self.encoder: nn.Module = NativeHeteroSAGE(
                data=data,
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_layers=number_of_layers,
            )

        elif architecture == "rgcn":
            number_of_bases = int(
                getattr(
                    cfg.model,
                    "num_bases",
                    8,
                )
            )

            self.encoder = ExplicitRGCN(
                data=data,
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_layers=number_of_layers,
                num_bases=number_of_bases,
            )

        elif architecture in MATCHED_ENCODER_NAMES:
            # Fair architecture family: shared projection + identical decoder width.
            metadata = (
                list(data.node_types),
                list(data.edge_types),
            )
            dict_encoder = build_taskA_encoder(cfg, metadata=metadata)
            self.encoder = FlatDictEncoderAdapter(dict_encoder)

        else:
            raise ValueError(
                "Unknown structural-reconstruction architecture: "
                f"{architecture!r}. "
                "Available architectures: hetero_sage, rgcn, "
                "hetero_sage_matched, hetero_gatv2, hgt."
            )

        hcr_cfg = getattr(cfg, "hcr", None)
        self.use_hcr = bool(
            hcr_cfg is not None and bool(getattr(hcr_cfg, "enabled", False))
        )
        self.hcr_dim = 0
        decoder_cfg = getattr(cfg.model, "decoder", None)
        decoder_dropout = float(
            getattr(decoder_cfg, "dropout", dropout)
            if decoder_cfg is not None
            else dropout
        )
        decoder_activation = str(
            getattr(decoder_cfg, "activation", "relu")
            if decoder_cfg is not None
            else "relu"
        )
        decoder_hidden_dims = None
        if decoder_cfg is not None and getattr(decoder_cfg, "hidden_dims", None) is not None:
            decoder_hidden_dims = [int(x) for x in list(decoder_cfg.hidden_dims)]
        elif decoder_cfg is not None and getattr(decoder_cfg, "hidden_dim", None) is not None:
            decoder_hidden_dims = [int(decoder_cfg.hidden_dim)]

        decoder_name = str(
            getattr(decoder_cfg, "name", "hcr_mlp") if decoder_cfg is not None else "hcr_mlp"
        ).strip().lower()

        if decoder_name in {
            "fusion88",
            "fusion_88",
            "graph_fusion88",
            "stage_a_fusion88",
            "fusion88_stat",
            "stage_c_fusion88",
        }:
            from taskA_final_large_grid_11_08_2026.fusion88_decoder import (
                build_fusion88_decoder,
            )
            from taskA_final_large_grid_11_08_2026.stage_c.stat_encoder import (
                Fusion88StatDecoder,
                build_stage_c_decoder,
            )

            # Stage C: Fusion88StatDecoder when stat_raw_dim / stat_variant set;
            # S0 stays on plain Fusion88Decoder (exact zeros g_stat path).
            wants_stat = decoder_name in {"fusion88_stat", "stage_c_fusion88"}
            if not wants_stat and decoder_cfg is not None:
                wants_stat = (
                    getattr(decoder_cfg, "stat_raw_dim", None) is not None
                    or bool(getattr(decoder_cfg, "force_zero_stat", False))
                )
            if not wants_stat:
                exp = getattr(cfg, "experiment", None)
                sv = str(getattr(exp, "stat_variant", "") or "").upper()
                wants_stat = bool(sv) or "STAGE_C" in str(
                    getattr(exp, "variant", "") or ""
                ).upper()

            if wants_stat:
                self.decoder = build_stage_c_decoder(cfg, hidden_channels)
                self.fusion88_stat = isinstance(self.decoder, Fusion88StatDecoder)
            else:
                self.decoder = build_fusion88_decoder(cfg, hidden_channels)
                self.fusion88_stat = False
            self.hcr_dim = 0
            self.residual_fusion = False
            self.wave7c = False
            self.fusion88 = True
            self.use_hcr = False
        elif self.use_hcr and decoder_name in {
            "wave7c",
            "wave7c_b2",
            "hcr_wave7c",
        }:
            from models.TaskA.wave7c_decoder import build_wave7c_decoder

            self.decoder = build_wave7c_decoder(cfg, hidden_channels)
            self.hcr_dim = int(getattr(self.decoder, "hcr_dim", 120))
            self.residual_fusion = False
            self.wave7c = True
            self.fusion88 = False
        elif self.use_hcr and decoder_name in {
            "residual_fusion",
            "hcr_residual_fusion",
            "panel_b_residual",
        }:
            from models.TaskA.residual_fusion_decoder import build_residual_fusion_decoder

            self.decoder = build_residual_fusion_decoder(cfg, hidden_channels)
            self.hcr_dim = int(getattr(self.decoder, "hcr_dim", 120))
            self.residual_fusion = True
            self.wave7c = False
            self.fusion88 = False
        elif self.use_hcr:
            if decoder_cfg is not None and getattr(decoder_cfg, "hcr_dim", None) is not None:
                self.hcr_dim = int(decoder_cfg.hcr_dim)
            else:
                self.hcr_dim = int(getattr(hcr_cfg, "output_dim", 8))
            self.decoder: nn.Module = HCRLinkDecoder(
                hidden_channels=hidden_channels,
                hcr_dim=self.hcr_dim,
                dropout=decoder_dropout,
                hidden_dims=decoder_hidden_dims,
                activation=decoder_activation,
            )
            self.residual_fusion = False
            self.wave7c = False
            self.fusion88 = False
        else:
            self.decoder = LinkDecoder(
                hidden_channels=hidden_channels,
                dropout=decoder_dropout,
            )
            self.residual_fusion = False
            self.wave7c = False
            self.fusion88 = False

        self.num_layers = number_of_layers

        self._node_types_sorted = sorted(
            data.node_types
        )

        self._node_sizes = {
            node_type: int(
                data[node_type].num_nodes
            )
            for node_type in self._node_types_sorted
        }

        self._layout_signature = layout_signature(
            data
        )

        training_cfg = getattr(
            cfg,
            "training",
            None,
        )

        if training_cfg is None:
            configured_grad_clip = 0.0
        else:
            configured_grad_clip = float(
                getattr(
                    training_cfg,
                    "grad_clip",
                    0.0,
                )
                or 0.0
            )

        self.grad_clip = (
            configured_grad_clip
            if configured_grad_clip > 0.0
            else None
        )

    def _validate_runtime_layout(
        self,
        data: Any,
    ) -> None:
        """Ensure train, validation and test use the same node layout."""

        current_signature = layout_signature(
            data
        )

        if current_signature != self._layout_signature:
            raise ValueError(
                "The runtime graph node layout differs from the layout "
                "used to initialize the model.\n"
                f"Expected: {self._layout_signature}\n"
                f"Received: {current_signature}"
            )

    def encode_flat(
        self,
        data: Any,
    ) -> torch.Tensor:
        """Return one global flat node-embedding matrix."""

        self._validate_runtime_layout(
            data
        )

        embeddings = self.encoder(
            data
        )

        expected_number_of_nodes = sum(
            self._node_sizes.values()
        )

        if embeddings.ndim != 2:
            raise ValueError(
                "Encoder must return a two-dimensional tensor. "
                f"Received shape {tuple(embeddings.shape)}."
            )

        if embeddings.size(0) != expected_number_of_nodes:
            raise ValueError(
                "Encoder returned an incorrect number of node embeddings. "
                f"Expected {expected_number_of_nodes}, "
                f"received {embeddings.size(0)}."
            )

        return embeddings

    def encode(
        self,
        data: Any,
    ) -> dict[str, torch.Tensor]:
        """Return embeddings separated by node type."""

        flat_embeddings = self.encode_flat(
            data
        )

        embedding_dict: dict[
            str,
            torch.Tensor,
        ] = {}

        start = 0

        for node_type in self._node_types_sorted:
            number_of_nodes = self._node_sizes[
                node_type
            ]

            end = (
                start
                + number_of_nodes
            )

            embedding_dict[node_type] = (
                flat_embeddings[
                    start:end
                ]
            )

            start = end

        return embedding_dict

    def forward(
        self,
        data: Any,
    ) -> torch.Tensor:
        """Encode G-TRAIN and return one logit per candidate pair."""

        flat_embeddings = self.encode_flat(
            data
        )

        if getattr(self, "fusion88", False):
            # Stage A: g_stat omitted → zeros(24).
            # Stage C: pass attached stat_raw [N,3,D] + role masks when present;
            # Fusion88StatDecoder encodes them; plain Fusion88Decoder ignores extras.
            return self.decoder(
                embeddings=flat_embeddings,
                source=data.link_source_idx,
                target=data.link_target_idx,
                g_stat=getattr(data, "g_stat", None),
                stat_raw=getattr(data, "stat_raw", None),
                stat_role_masks=getattr(data, "stat_role_masks", None),
            )

        if self.use_hcr:
            hcr_features = getattr(data, "hcr_features", None)
            if hcr_features is None:
                raise RuntimeError(
                    "HCR is enabled but data.hcr_features is missing. "
                    "Call fit_and_attach_hcr(...) before training."
                )
            if getattr(self, "wave7c", False):
                return self.decoder(
                    embeddings=flat_embeddings,
                    source=data.link_source_idx,
                    target=data.link_target_idx,
                    hcr_features=hcr_features,
                    hcr_role_masks=getattr(data, "hcr_role_masks", None),
                    hcr_triple_features=getattr(data, "hcr_triple_features", None),
                    hcr_nonbinary_mask=getattr(data, "hcr_nonbinary_mask", None),
                )
            if getattr(self, "residual_fusion", False):
                legacy = getattr(data, "hcr_legacy_features", None)
                if getattr(self.decoder, "legacy_mode", "") == "classical4":
                    legacy = getattr(data, "hcr_classical4_features", legacy)
                return self.decoder(
                    embeddings=flat_embeddings,
                    source=data.link_source_idx,
                    target=data.link_target_idx,
                    hcr_features=hcr_features,
                    hcr_legacy_features=legacy,
                    hcr_bb_mask=getattr(data, "hcr_bb_mask", None),
                    hcr_triple_features=getattr(data, "hcr_triple_features", None),
                )
            return self.decoder(
                embeddings=flat_embeddings,
                source=data.link_source_idx,
                target=data.link_target_idx,
                hcr_features=hcr_features,
            )

        return self.decoder(
            embeddings=flat_embeddings,
            source=data.link_source_idx,
            target=data.link_target_idx,
        )