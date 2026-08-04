"""K2 — MLP then KAN: Linear(40→16)→GELU→LN→Drop→KANLinear(16→8)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from .kan_linear import KANLinear

if TYPE_CHECKING:
    from .factory import PairEncoderConfig


class MLPToKANPairEncoder(nn.Module):
    def __init__(self, config: "PairEncoderConfig") -> None:
        super().__init__()
        if config.input_dim != 40 or config.output_dim != 8:
            raise ValueError(
                f"MLP→KAN expects 40→8, got {config.input_dim}→{config.output_dim}"
            )
        h = int(config.hidden_dim)
        drop = float(config.dropout)
        grid_range = tuple(float(v) for v in config.grid_range)
        self.proj = nn.Sequential(
            nn.Linear(config.input_dim, h),
            nn.GELU(),
            nn.LayerNorm(h),
            nn.Dropout(drop),
        )
        self.kan = KANLinear(
            h,
            config.output_dim,
            grid_size=int(config.grid_size),
            spline_order=int(config.spline_order),
            grid_range=grid_range,
            grid_update=bool(config.grid_update),
            base_activation=str(config.base_activation),
            base_scale_init=float(config.base_scale_init),
            spline_scale_init=float(config.spline_scale_init),
            use_bias=bool(config.use_bias),
        )
        self.norm_out = nn.LayerNorm(config.output_dim)
        self.spline_l1 = float(config.spline_l1)
        self.grid_update = bool(config.grid_update)

    def forward(self, pair_features: torch.Tensor) -> torch.Tensor:
        if pair_features.ndim != 2 or pair_features.size(-1) != 40:
            raise ValueError(f"expected [B,40], got {tuple(pair_features.shape)}")
        h = self.proj(pair_features)
        if self.training and self.grid_update:
            self.kan.update_grid_from_data(h)
        return self.norm_out(self.kan(h))

    def regularization_loss(self) -> torch.Tensor:
        zero = self.kan.spline_weight.new_zeros(())
        if self.spline_l1 <= 0:
            return zero
        return self.spline_l1 * self.kan.regularization_loss()

    @torch.no_grad()
    def diagnostics(self, x: torch.Tensor | None = None) -> dict[str, float]:
        h = None if x is None else self.proj(x)
        d = {f"kan/{k}": v for k, v in self.kan.diagnostics(h).items()}
        d["pair_encoder_parameter_count"] = float(
            sum(p.numel() for p in self.parameters())
        )
        return d
