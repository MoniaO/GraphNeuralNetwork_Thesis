"""K5 — grouped KAN: numeric[0:32]→KAN(→8), metadata[32:40]→MLP(→4), fuse→8."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from .kan_linear import KANLinear

if TYPE_CHECKING:
    from .factory import PairEncoderConfig

NUMERIC_DIM = 32
METADATA_DIM = 8
META_LATENT = 4


class GroupedKANPairEncoder(nn.Module):
    def __init__(self, config: "PairEncoderConfig") -> None:
        super().__init__()
        if config.input_dim != 40 or config.output_dim != 8:
            raise ValueError(
                f"Grouped KAN expects 40→8, got {config.input_dim}→{config.output_dim}"
            )
        grid_range = tuple(float(v) for v in config.grid_range)
        drop = float(config.dropout)
        self.kan = KANLinear(
            NUMERIC_DIM,
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
        self.meta_mlp = nn.Sequential(
            nn.Linear(METADATA_DIM, META_LATENT),
            nn.GELU(),
            nn.LayerNorm(META_LATENT),
            nn.Dropout(drop),
        )
        self.fuse = nn.Linear(config.output_dim + META_LATENT, config.output_dim)
        self.norm_out = nn.LayerNorm(config.output_dim)
        self.spline_l1 = float(config.spline_l1)
        self.grid_update = bool(config.grid_update)
        self.last_branch_stats: dict[str, float] = {}

    def forward(self, pair_features: torch.Tensor) -> torch.Tensor:
        if pair_features.ndim != 2 or pair_features.size(-1) != 40:
            raise ValueError(f"expected [B,40], got {tuple(pair_features.shape)}")
        numeric = pair_features[:, :NUMERIC_DIM]
        metadata = pair_features[:, NUMERIC_DIM:]
        if self.training and self.grid_update:
            self.kan.update_grid_from_data(numeric)
        u = self.kan(numeric)
        m = self.meta_mlp(metadata)
        with torch.no_grad():
            un = float(u.norm(dim=-1).mean())
            mn = float(m.norm(dim=-1).mean())
            self.last_branch_stats = {
                "numeric_branch_norm": un,
                "metadata_branch_norm": mn,
                "numeric_to_metadata_ratio": un / max(mn, 1e-8),
            }
        return self.norm_out(self.fuse(torch.cat([u, m], dim=-1)))

    def regularization_loss(self) -> torch.Tensor:
        zero = self.kan.spline_weight.new_zeros(())
        if self.spline_l1 <= 0:
            return zero
        return self.spline_l1 * self.kan.regularization_loss()

    @torch.no_grad()
    def diagnostics(self, x: torch.Tensor | None = None) -> dict[str, float]:
        xn = None if x is None else x[:, :NUMERIC_DIM]
        d = {f"kan/{k}": v for k, v in self.kan.diagnostics(xn).items()}
        d.update(self.last_branch_stats)
        d["pair_encoder_parameter_count"] = float(
            sum(p.numel() for p in self.parameters())
        )
        return d
