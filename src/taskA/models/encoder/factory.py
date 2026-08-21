"""Task A encoder factory (HGT / SAGE / GATv2 / RGCN).

What it does
------------
Reads `cfg.model.encoder_name` (or conv_type) and builds the matched encoder.
FINAL always selects `hgt`. SAGE/GAT/RGCN exist only from the Stage A race.

What you may change
-------------------
Backbone name and hypers in yaml / in the Stage A runner.
Do not add decoder logic here — the decoder lives in `models.decoder`.
"""

from __future__ import annotations

from typing import Any

from .base import BaseHeteroEncoder
from .gatv2 import HeteroGATv2Encoder
from .sage import HeteroSAGEMatchedEncoder
from .hgt import HGTEncoder
from .input_projection import HeteroInputProjection


def build_taskA_encoder(cfg: Any, metadata: tuple) -> BaseHeteroEncoder:
    """Build a matched-family Task A encoder (SAGE / GATv2 / HGT)."""
    name = str(
        getattr(cfg.model, "encoder_name", None)
        or getattr(cfg.model, "conv_type", None)
        or getattr(cfg.model, "name", "")
    ).strip().lower()

    hidden_dim = int(
        getattr(cfg.model, "hidden_dim", None)
        or getattr(cfg.model, "hidden_channels", 64)
    )
    common = {
        "metadata": metadata,
        "hidden_dim": hidden_dim,
        "num_layers": int(getattr(cfg.model, "num_layers", 2)),
        "dropout": float(getattr(cfg.model, "dropout", 0.2)),
        "residual": bool(getattr(cfg.model, "residual", True)),
    }

    if name in {"hetero_sage_matched", "sage_matched"}:
        return HeteroSAGEMatchedEncoder(
            **common,
            aggr=str(getattr(cfg.model, "relation_aggr", "sum")),
        )

    if name in {"hetero_gatv2", "gatv2"}:
        return HeteroGATv2Encoder(
            **common,
            heads=int(getattr(cfg.model, "heads", 4)),
            aggr=str(getattr(cfg.model, "relation_aggr", "sum")),
        )

    if name == "hgt":
        hgt_cfg = getattr(cfg.model, "hgt", None)
        activation = str(
            getattr(hgt_cfg, "activation", None)
            or getattr(cfg.model, "activation", "relu")
        )
        heads = int(
            getattr(hgt_cfg, "heads", None)
            or getattr(cfg.model, "heads", 4)
        )
        if hgt_cfg is not None:
            common = {
                **common,
                "hidden_dim": int(
                    getattr(hgt_cfg, "hidden_dim", common["hidden_dim"])
                ),
                "num_layers": int(
                    getattr(hgt_cfg, "num_layers", common["num_layers"])
                ),
                "dropout": float(
                    getattr(hgt_cfg, "dropout", common["dropout"])
                ),
                "residual": bool(
                    getattr(hgt_cfg, "residual", common["residual"])
                ),
            }
        return HGTEncoder(
            **common,
            heads=heads,
            activation=activation,
        )

    if name in {"rgcn_matched", "hetero_rgcn_matched"}:
        from taskA.models.encoder.rgcn import (
            HeteroRGCNMatchedEncoder,
        )

        bases_raw = getattr(cfg.model, "num_bases", 8)
        return HeteroRGCNMatchedEncoder(
            **common,
            num_bases=bases_raw,
        )

    raise ValueError(f"Unknown Task A matched encoder: {name}")


__all__ = [
    "BaseHeteroEncoder",
    "HeteroInputProjection",
    "HeteroSAGEMatchedEncoder",
    "HeteroGATv2Encoder",
    "HGTEncoder",
    "build_taskA_encoder",
]
