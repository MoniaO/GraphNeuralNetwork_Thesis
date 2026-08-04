"""K4 — linear skip + gated KAN correction (full retrain, not Wave 10 freeze)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from .kan_linear import KANLinear

if TYPE_CHECKING:
    from .factory import PairEncoderConfig


class LinearPlusKANPairEncoder(nn.Module):
    def __init__(self, config: "PairEncoderConfig") -> None:
        super().__init__()
        if config.input_dim != 40 or config.output_dim != 8:
            raise ValueError(
                f"Linear+KAN expects 40→8, got {config.input_dim}→{config.output_dim}"
            )
        grid_range = tuple(float(v) for v in config.grid_range)
        self.linear = nn.Linear(config.input_dim, config.output_dim)
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
        gate0 = float(getattr(config, "gate_init_logit", -3.0))
        self.gate_logit = nn.Parameter(torch.tensor(gate0, dtype=torch.float32))
        self.spline_l1 = float(config.spline_l1)
        self.grid_update = bool(config.grid_update)
        self.last_gate_stats: dict[str, float] = {}

    def forward(self, pair_features: torch.Tensor) -> torch.Tensor:
        if pair_features.ndim != 2 or pair_features.size(-1) != 40:
            raise ValueError(f"expected [B,40], got {tuple(pair_features.shape)}")
        if self.training and self.grid_update:
            self.kan.update_grid_from_data(pair_features)
        lin = self.linear(pair_features)
        kan = self.kan(pair_features)
        alpha = torch.sigmoid(self.gate_logit)
        out = lin + alpha * kan
        with torch.no_grad():
            ln = float(lin.norm(dim=-1).mean())
            kn = float(kan.norm(dim=-1).mean())
            self.last_gate_stats = {
                "alpha": float(alpha.detach()),
                "linear_output_norm": ln,
                "kan_output_norm": kn,
                "kan_to_linear_ratio": kn / max(ln, 1e-8),
            }
        return out

    def regularization_loss(self) -> torch.Tensor:
        zero = self.kan.spline_weight.new_zeros(())
        if self.spline_l1 <= 0:
            return zero
        return self.spline_l1 * self.kan.regularization_loss()

    @torch.no_grad()
    def diagnostics(self, x: torch.Tensor | None = None) -> dict[str, float]:
        d = {f"kan/{k}": v for k, v in self.kan.diagnostics(x).items()}
        d.update(self.last_gate_stats)
        d["gate_logit"] = float(self.gate_logit.detach())
        d["pair_encoder_parameter_count"] = float(
            sum(p.numel() for p in self.parameters())
        )
        return d
