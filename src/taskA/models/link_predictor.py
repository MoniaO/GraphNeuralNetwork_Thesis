"""Link predictor: graph encoder + Fusion88 decoder (MLP/KAN).

What it does
------------
Wires `build_taskA_encoder` to Fusion88 / Fusion88Stat.
Forward: embeddings from G_train → logit for each candidate pair.

What you may change
-------------------
Decoder: `model.decoder.name` = fusion88 | fusion88_stat.
Pair encoder: `model.decoder.stat_pair_encoder` = mlp | kan_shallow.

What not to touch for FINAL
---------------------------
fusion88_stat + S10; encoder_name=hgt. Do not mix SAGE with Fusion88-stat
if you want to compare against the 14.08 table.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from taskA.models.encoder.factory import build_taskA_encoder
from taskA.data.layout import flatten_embeddings, layout_signature


MATCHED_ENCODER_NAMES = {
    "hetero_sage_matched",
    "sage_matched",
    "hetero_gatv2",
    "gatv2",
    "hgt",
    "rgcn_matched",
    "hetero_rgcn_matched",
}

FUSION88_NAMES = {
    "fusion88",
    "fusion_88",
    "graph_fusion88",
    "stage_a_fusion88",
    "fusion88_stat",
    "stage_c_fusion88",
}


class FlatDictEncoderAdapter(nn.Module):
    """Adapt a z_dict encoder to the flat embedding interface used by Fusion88."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, data: Any) -> torch.Tensor:
        embedding_dict = self.encoder(data)
        return flatten_embeddings(data=data, embedding_dict=embedding_dict)

    def encode_dict(self, data: Any) -> dict[str, torch.Tensor]:
        return self.encoder(data)


class HeteroReconGNN(nn.Module):
    """Task A binary link-prediction model (FINAL 14.08 stack)."""

    def __init__(
        self,
        cfg: Any,
        data: Any,
        in_channels: int,
        hidden_channels: int,
    ) -> None:
        super().__init__()
        del in_channels  # matched encoders infer width from HeteroData.x

        configured_architecture = getattr(cfg.model, "conv_type", None)
        if configured_architecture is None:
            configured_architecture = getattr(cfg.model, "name", None)
        if configured_architecture is None:
            raise ValueError("Set cfg.model.conv_type or cfg.model.name.")

        architecture = str(configured_architecture).strip().lower()

        hgt_cfg = getattr(cfg.model, "hgt", None)
        if hgt_cfg is not None:
            if getattr(hgt_cfg, "hidden_dim", None) is not None:
                hidden_channels = int(hgt_cfg.hidden_dim)
            if getattr(hgt_cfg, "num_layers", None) is not None:
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

        number_of_layers = int(getattr(cfg.model, "num_layers", 2))
        dropout = float(getattr(cfg.model, "dropout", 0.20))

        if architecture not in MATCHED_ENCODER_NAMES:
            raise ValueError(
                "Unknown architecture: "
                f"{architecture!r}. "
                "Available: hgt, hetero_sage_matched, hetero_gatv2, rgcn_matched."
            )

        metadata = (list(data.node_types), list(data.edge_types))
        dict_encoder = build_taskA_encoder(cfg, metadata=metadata)
        self.encoder = FlatDictEncoderAdapter(dict_encoder)

        decoder_cfg = getattr(cfg.model, "decoder", None)
        decoder_name = str(
            getattr(decoder_cfg, "name", "fusion88_stat")
            if decoder_cfg is not None
            else "fusion88_stat"
        ).strip().lower()
        if decoder_name not in FUSION88_NAMES:
            raise ValueError(
                f"Unknown decoder {decoder_name!r}. "
                "FINAL stack uses fusion88 (Stage A zeros) or fusion88_stat (S10)."
            )

        from taskA.models.decoder.fusion88 import (
            build_fusion88_decoder,
        )
        from taskA.models.decoder.pair_encoder import (
            Fusion88StatDecoder,
            build_stage_c_decoder,
        )

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
        self.fusion88 = True
        self.num_layers = number_of_layers
        self._node_types_sorted = sorted(data.node_types)
        self._node_sizes = {
            node_type: int(data[node_type].num_nodes)
            for node_type in self._node_types_sorted
        }
        self._layout_signature = layout_signature(data)

        training_cfg = getattr(cfg, "training", None)
        configured_grad_clip = (
            float(getattr(training_cfg, "grad_clip", 0.0) or 0.0)
            if training_cfg is not None
            else 0.0
        )
        self.grad_clip = configured_grad_clip if configured_grad_clip > 0.0 else None
        del dropout

    def _validate_runtime_layout(self, data: Any) -> None:
        current_signature = layout_signature(data)
        if current_signature != self._layout_signature:
            raise ValueError(
                "The runtime graph node layout differs from the layout "
                "used to initialize the model.\n"
                f"Expected: {self._layout_signature}\n"
                f"Received: {current_signature}"
            )

    def encode_flat(self, data: Any) -> torch.Tensor:
        self._validate_runtime_layout(data)
        embeddings = self.encoder(data)
        expected_number_of_nodes = sum(self._node_sizes.values())
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

    def encode(self, data: Any) -> dict[str, torch.Tensor]:
        flat_embeddings = self.encode_flat(data)
        embedding_dict: dict[str, torch.Tensor] = {}
        start = 0
        for node_type in self._node_types_sorted:
            number_of_nodes = self._node_sizes[node_type]
            end = start + number_of_nodes
            embedding_dict[node_type] = flat_embeddings[start:end]
            start = end
        return embedding_dict

    def forward(self, data: Any) -> torch.Tensor:
        flat_embeddings = self.encode_flat(data)
        return self.decoder(
            embeddings=flat_embeddings,
            source=data.link_source_idx,
            target=data.link_target_idx,
            g_stat=getattr(data, "g_stat", None),
            stat_raw=getattr(data, "stat_raw", None),
            stat_role_masks=getattr(data, "stat_role_masks", None),
        )
