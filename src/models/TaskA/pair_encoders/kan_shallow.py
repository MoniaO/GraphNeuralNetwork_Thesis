"""K1 — shallow KAN: KANLinear(40 → 8)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from .kan_linear import KANLinear

if TYPE_CHECKING:
    from .factory import PairEncoderConfig


class KANShallowPairEncoder(nn.Module):
    def __init__(self, config: "PairEncoderConfig") -> None:
        super().__init__()
        if config.input_dim != 40 or config.output_dim != 8:
            raise ValueError(
                f"KAN shallow expects 40→8, got {config.input_dim}→{config.output_dim}"
            )
        grid_range = tuple(float(v) for v in config.grid_range)
        self.kan = KANLinear(
            config.input_dim,
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
        if self.training and self.grid_update:
            self.kan.update_grid_from_data(pair_features)
        return self.norm_out(self.kan(pair_features))

    def regularization_loss(self) -> torch.Tensor:
        zero = self.kan.spline_weight.new_zeros(())
        if self.spline_l1 <= 0:
            return zero
        return self.spline_l1 * self.kan.regularization_loss()

    @torch.no_grad()
    def diagnostics(self, x: torch.Tensor | None = None) -> dict[str, float]:
        d = self.kan.diagnostics(x)
        d["pair_encoder_parameter_count"] = float(
            sum(p.numel() for p in self.parameters())
        )
        return d
