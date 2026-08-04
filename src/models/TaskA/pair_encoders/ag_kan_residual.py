"""Wave 10 — small AG-only KAN residual over frozen MLP latent.

g*_AG = g_AG_MLP + α · r_AG_KAN
α = σ(gate_logit), gate_logit init = −3 → α ≈ 0.047
r_AG_KAN = KANLinear(40 → 8)  (single layer; correction, not full encoder)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
import torch.nn as nn

from .kan_linear import KANLinear


@dataclass
class AGKANResidualConfig:
    enabled: bool = False
    input_dim: int = 40
    output_dim: int = 8
    spline_order: int = 3
    grid_size: int = 3
    grid_range: Sequence[float] = field(default_factory=lambda: (-3.0, 3.0))
    grid_update: bool = False
    base_activation: str = "silu"
    base_scale_init: float = 0.1
    spline_scale_init: float = 0.05
    use_bias: bool = True
    gate_init_logit: float = -3.0
    spline_l1: float = 1.0e-5
    type_routed: bool = False  # T1: zero residual on binary–binary
    # train_mode: residual_only | finetune (handled outside)
    train_mode: str = "residual_only"

    @classmethod
    def from_mapping(cls, raw: Any) -> "AGKANResidualConfig":
        if raw is None:
            return cls()
        if isinstance(raw, AGKANResidualConfig):
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


class AGKANResidual(nn.Module):
    """Single-layer KAN residual + scalar sigmoid gate for AG role only."""

    def __init__(self, config: AGKANResidualConfig | None = None) -> None:
        super().__init__()
        cfg = config or AGKANResidualConfig(enabled=True)
        self.config = cfg
        self.type_routed = bool(cfg.type_routed)
        self.spline_l1 = float(cfg.spline_l1)
        self.kan = KANLinear(
            int(cfg.input_dim),
            int(cfg.output_dim),
            grid_size=int(cfg.grid_size),
            spline_order=int(cfg.spline_order),
            grid_range=tuple(cfg.grid_range),
            grid_update=bool(cfg.grid_update),
            base_activation=str(cfg.base_activation),
            base_scale_init=float(cfg.base_scale_init),
            spline_scale_init=float(cfg.spline_scale_init),
            use_bias=bool(cfg.use_bias),
        )
        self.gate_logit = nn.Parameter(
            torch.tensor([float(cfg.gate_init_logit)], dtype=torch.float32)
        )
        self.last_stats: dict[str, float] = {}

    def forward(
        self,
        h_ag: torch.Tensor,
        *,
        role_mask: torch.Tensor | None = None,
        nonbinary_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (gated_residual, alpha, raw_residual).

        role_mask / nonbinary_mask: [B] or [B,1] — multiply residual before gate scale.
        Unavailable AG → exact zero residual.
        """
        raw_total, base_out, spline_out = self.kan.forward_parts(h_ag)
        raw = raw_total
        if role_mask is not None:
            m = role_mask.to(device=raw.device, dtype=raw.dtype).view(-1, 1)
            raw = raw * m
            base_out = base_out * m
            spline_out = spline_out * m
        if self.type_routed and nonbinary_mask is not None:
            nb = nonbinary_mask.to(device=raw.device, dtype=raw.dtype).view(-1, 1)
            raw = raw * nb
            base_out = base_out * nb
            spline_out = spline_out * nb
        alpha = torch.sigmoid(self.gate_logit)
        gated = alpha * raw
        with torch.no_grad():
            self.last_stats = {
                "gate_alpha": float(alpha.detach()),
                "kan_residual_norm": float(raw.detach().norm(dim=-1).mean()),
                "gated_residual_norm": float(gated.detach().norm(dim=-1).mean()),
                "gate_logit": float(self.gate_logit.detach()),
                "base_path_norm": float(base_out.detach().norm(dim=-1).mean()),
                "spline_path_norm": float(spline_out.detach().norm(dim=-1).mean()),
            }
        return gated, alpha, raw

    def regularization_loss(self) -> torch.Tensor:
        return self.spline_l1 * self.kan.regularization_loss()

    def trainable_parameters(self):
        return list(self.parameters())
