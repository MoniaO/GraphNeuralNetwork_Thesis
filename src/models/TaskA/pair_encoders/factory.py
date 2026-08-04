"""Factory for Task A pair encoders (MLP control + KAN architecture audit)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import torch.nn as nn

from .mlp import MLPPairEncoder

EncoderType = Literal[
    "mlp",
    "kan",
    "kan_shallow",
    "mlp_to_kan",
    "kan_to_linear",
    "linear_plus_kan",
    "grouped_kan",
]


@dataclass
class PairEncoderConfig:
    type: EncoderType = "mlp"
    input_dim: int = 40
    hidden_dim: int = 16
    output_dim: int = 8
    dropout: float = 0.1
    # MLP
    activation: str = "gelu"
    layernorm: bool = True
    # KAN
    spline_order: int = 3
    grid_size: int = 5
    grid_range: Sequence[float] = field(default_factory=lambda: (-3.0, 3.0))
    grid_update: bool = False
    base_activation: str = "silu"
    base_scale_init: float = 1.0
    spline_scale_init: float = 0.1
    use_bias: bool = True
    spline_l1: float = 1.0e-5
    # K4
    gate_init_logit: float = -3.0

    @classmethod
    def from_mapping(cls, raw: Any) -> "PairEncoderConfig":
        if raw is None:
            return cls()
        if isinstance(raw, cls):
            return raw
        try:
            from omegaconf import OmegaConf

            if OmegaConf.is_config(raw):
                d = OmegaConf.to_container(raw, resolve=True)
            else:
                d = dict(raw) if not isinstance(raw, dict) else raw
        except Exception:
            d = dict(raw) if not isinstance(raw, dict) else raw
        assert isinstance(d, dict)
        kwargs = {k: d[k] for k in cls.__dataclass_fields__ if k in d}  # type: ignore[attr-defined]
        if "grid_range" in kwargs and kwargs["grid_range"] is not None:
            kwargs["grid_range"] = tuple(float(v) for v in kwargs["grid_range"])
        return cls(**kwargs)


def build_pair_encoder(config: Any = None) -> nn.Module:
    """Return a PairEncoder: [B,40] → [B,8]."""
    cfg = PairEncoderConfig.from_mapping(config)
    t = str(cfg.type).lower()
    if t == "mlp":
        return MLPPairEncoder(cfg)
    if t in {"kan", "kan_direct"}:
        from .kan import KANPairEncoder

        return KANPairEncoder(cfg)
    if t == "kan_shallow":
        from .kan_shallow import KANShallowPairEncoder

        return KANShallowPairEncoder(cfg)
    if t == "mlp_to_kan":
        from .mlp_to_kan import MLPToKANPairEncoder

        return MLPToKANPairEncoder(cfg)
    if t == "kan_to_linear":
        from .kan_to_linear import KANToLinearPairEncoder

        return KANToLinearPairEncoder(cfg)
    if t == "linear_plus_kan":
        from .linear_plus_kan import LinearPlusKANPairEncoder

        return LinearPlusKANPairEncoder(cfg)
    if t == "grouped_kan":
        from .grouped_kan import GroupedKANPairEncoder

        return GroupedKANPairEncoder(cfg)
    raise ValueError(f"Unknown pair encoder: {cfg.type}")
