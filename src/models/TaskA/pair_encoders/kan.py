"""KAN pair encoder: 40→16→8 with LayerNorm + Dropout (no external GELU)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from .kan_linear import KANLinear

if TYPE_CHECKING:
    from .factory import PairEncoderConfig


class KANPairEncoder(nn.Module):
    """Wave 9 K1 stack matching MLP geometry without post-KAN GELU."""

    def __init__(self, config: "PairEncoderConfig") -> None:
        super().__init__()
        if config.input_dim != 40 or config.output_dim != 8:
            raise ValueError(
                f"Wave9 KAN expects 40→8, got {config.input_dim}→{config.output_dim}"
            )
        grid_range = tuple(float(v) for v in config.grid_range)
        common = dict(
            grid_size=int(config.grid_size),
            spline_order=int(config.spline_order),
            grid_range=grid_range,
            grid_update=bool(config.grid_update),
            base_activation=str(config.base_activation),
            base_scale_init=float(config.base_scale_init),
            spline_scale_init=float(config.spline_scale_init),
            use_bias=bool(config.use_bias),
        )
        self.layer1 = KANLinear(config.input_dim, int(config.hidden_dim), **common)
        self.norm = nn.LayerNorm(int(config.hidden_dim))
        self.dropout = nn.Dropout(float(config.dropout))
        self.layer2 = KANLinear(int(config.hidden_dim), config.output_dim, **common)
        self.norm_out = nn.LayerNorm(config.output_dim)
        self.spline_l1 = float(config.spline_l1)
        self.grid_update = bool(config.grid_update)

    def forward(self, pair_features: torch.Tensor) -> torch.Tensor:
        if pair_features.ndim != 2 or pair_features.size(-1) != 40:
            raise ValueError(f"expected [B,40], got {tuple(pair_features.shape)}")
        if self.training and self.grid_update:
            self.layer1.update_grid_from_data(pair_features)
        x = self.layer1(pair_features)
        x = self.norm(x)
        x = self.dropout(x)
        if self.training and self.grid_update:
            self.layer2.update_grid_from_data(x)
        x = self.layer2(x)
        x = self.norm_out(x)
        return x

    def regularization_loss(self) -> torch.Tensor:
        zero = self.layer1.spline_weight.new_zeros(())
        if self.spline_l1 <= 0:
            return zero
        return self.spline_l1 * (
            self.layer1.regularization_loss() + self.layer2.regularization_loss()
        )

    @torch.no_grad()
    def diagnostics(self, x: torch.Tensor | None = None) -> dict[str, float]:
        d1 = self.layer1.diagnostics(x)
        # After layer1+norm path for layer2 out-of-grid, use layer1 output if x given.
        x2 = None
        if x is not None:
            x2 = self.norm(self.layer1(x))
        d2 = self.layer2.diagnostics(x2)
        out = {f"layer1/{k}": v for k, v in d1.items()}
        out.update({f"layer2/{k}": v for k, v in d2.items()})
        out["pair_encoder_parameter_count"] = float(
            sum(p.numel() for p in self.parameters())
        )
        return out
