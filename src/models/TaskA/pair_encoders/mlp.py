"""Exact A1 MLP pair encoder: 40→16→8 with GELU + LayerNorm + Dropout."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .factory import PairEncoderConfig


class MLPPairEncoder(nn.Module):
    """Bit-compatible with ``wave7c_decoder._mlp_pair_encoder``."""

    def __init__(self, config: "PairEncoderConfig") -> None:
        super().__init__()
        if config.input_dim != 40 or config.output_dim != 8:
            raise ValueError(
                f"Wave9 MLP control expects 40→8, got "
                f"{config.input_dim}→{config.output_dim}"
            )
        hidden = int(config.hidden_dim)
        drop = float(config.dropout)
        layers: list[nn.Module] = [nn.Linear(config.input_dim, hidden)]
        act = str(config.activation).lower()
        if act == "gelu":
            layers.append(nn.GELU())
        elif act == "silu":
            layers.append(nn.SiLU())
        else:
            raise ValueError(f"unsupported MLP activation {config.activation}")
        if config.layernorm:
            layers.append(nn.LayerNorm(hidden))
        layers.append(nn.Dropout(drop))
        layers.append(nn.Linear(hidden, config.output_dim))
        if config.layernorm:
            layers.append(nn.LayerNorm(config.output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, pair_features: torch.Tensor) -> torch.Tensor:
        if pair_features.ndim != 2 or pair_features.size(-1) != 40:
            raise ValueError(
                f"expected [B,40], got {tuple(pair_features.shape)}"
            )
        return self.net(pair_features)
