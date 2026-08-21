"""Pair encoder (MLP vs KAN) + Fusion88StatDecoder.

What it does
------------
Three independent role encoders AZ / AG / ZG read raw S10 features (D=40).
Each role → 8-D latent; concat → g_stat ∈ R^24.
The graph branch and fusion head are identical to Fusion88Decoder.

MLP (FINAL default): D → 16 → GELU → LN → Drop 0.1 → 8 → LN
KAN (twin):          KANLinear(D→8) + LN  — same latent, fair comparison

What you may change (new experiment)
------------------------------------
- `model.decoder.stat_pair_encoder` = mlp | kan_shallow
- `model.decoder.pair_dropout` (MLP) and `spline_l1` (KAN)
- `stat_raw_dim` must match the variant (S10 = 40)

What not to touch for FINAL 14.08
---------------------------------
PAIR_LATENT=8, PAIR_HIDDEN=16, STAT_DIM=24, three unshared role heads.
Do not share weights across roles or change the KAN latent width.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from taskA.models.decoder.kan_linear import KANLinear
from taskA.models.decoder.fusion88 import (
    FUSION_DIM,
    GRAPH_MID,
    GRAPH_OUT,
    STAT_DIM,
    Fusion88Decoder,
)

N_ROLES = 3
PAIR_LATENT = 8
PAIR_HIDDEN = 16
PAIR_DROPOUT = 0.1
KAN_SPLINE_L1 = 1.0e-5


class StatPairEncoder(nn.Module):
    """D→16→GELU→LN→Drop0.1→8→LN; supports variable D (not only 40)."""

    def __init__(self, input_dim: int, *, dropout: float = PAIR_DROPOUT) -> None:
        super().__init__()
        d = int(input_dim)
        if d < 1:
            raise ValueError(f"input_dim must be >= 1, got {d}")
        self.input_dim = d
        self.net = nn.Sequential(
            nn.Linear(d, PAIR_HIDDEN),
            nn.GELU(),
            nn.LayerNorm(PAIR_HIDDEN),
            nn.Dropout(float(dropout)),
            nn.Linear(PAIR_HIDDEN, PAIR_LATENT),
            nn.LayerNorm(PAIR_LATENT),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.size(-1) != self.input_dim:
            raise ValueError(
                f"StatPairEncoder expected [B,{self.input_dim}], got {tuple(x.shape)}"
            )
        return self.net(x)

    def regularization_loss(self) -> torch.Tensor:
        return next(self.parameters()).new_zeros(())


class StatKANPairEncoder(nn.Module):
    """KAN-shallow twin: KANLinear(D→8) + LN(8); grid frozen (no update)."""

    def __init__(
        self,
        input_dim: int,
        *,
        spline_l1: float = KAN_SPLINE_L1,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-3.0, 3.0),
    ) -> None:
        super().__init__()
        d = int(input_dim)
        if d < 1:
            raise ValueError(f"input_dim must be >= 1, got {d}")
        self.input_dim = d
        self.spline_l1 = float(spline_l1)
        self.kan = KANLinear(
            d,
            PAIR_LATENT,
            grid_size=int(grid_size),
            spline_order=int(spline_order),
            grid_range=grid_range,
            grid_update=False,
            base_activation="silu",
            base_scale_init=1.0,
            spline_scale_init=0.1,
            use_bias=True,
        )
        self.norm_out = nn.LayerNorm(PAIR_LATENT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.size(-1) != self.input_dim:
            raise ValueError(
                f"StatKANPairEncoder expected [B,{self.input_dim}], got {tuple(x.shape)}"
            )
        return self.norm_out(self.kan(x))

    def regularization_loss(self) -> torch.Tensor:
        zero = self.kan.spline_weight.new_zeros(())
        if self.spline_l1 <= 0:
            return zero
        return self.spline_l1 * self.kan.regularization_loss()


def _make_role_encoder(
    kind: str,
    input_dim: int,
    *,
    pair_dropout: float,
    spline_l1: float,
) -> nn.Module:
    k = str(kind).lower().strip()
    if k in {"mlp", "stat_mlp", ""}:
        return StatPairEncoder(input_dim, dropout=pair_dropout)
    if k in {"kan_shallow", "kan", "stat_kan"}:
        return StatKANPairEncoder(input_dim, spline_l1=spline_l1)
    raise ValueError(f"unknown stat_pair_encoder={kind!r}")


class Fusion88StatDecoder(nn.Module):
    """Fusion88 graph branch + unshared statistical role encoders → 88 → logit.

    S0 / force_zero_stat: g_stat = zeros(24) (identical to Stage A path).
    """

    def __init__(
        self,
        hidden_channels: int,
        *,
        stat_raw_dim: int = 1,
        dropout: float = 0.2,
        pair_dropout: float = PAIR_DROPOUT,
        force_zero_stat: bool = False,
        stat_pair_encoder: str = "mlp",
        spline_l1: float = KAN_SPLINE_L1,
    ) -> None:
        super().__init__()
        if hidden_channels < 1:
            raise ValueError(f"hidden_channels must be positive, got {hidden_channels}")
        self.hidden_channels = int(hidden_channels)
        self.hgt_in = 4 * self.hidden_channels
        self.stat_dim = STAT_DIM
        self.fusion_dim = FUSION_DIM
        self.stat_raw_dim = max(int(stat_raw_dim), 1)
        self.force_zero_stat = bool(force_zero_stat)
        self.stat_pair_encoder = str(stat_pair_encoder)

        self.graph_branch = nn.Sequential(
            nn.Linear(self.hgt_in, GRAPH_MID),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(GRAPH_MID, GRAPH_OUT),
            nn.GELU(),
            nn.LayerNorm(GRAPH_OUT),
        )
        self.role_encoders = nn.ModuleList(
            [
                _make_role_encoder(
                    self.stat_pair_encoder,
                    self.stat_raw_dim,
                    pair_dropout=pair_dropout,
                    spline_l1=spline_l1,
                )
                for _ in range(N_ROLES)
            ]
        )
        self.output_head = nn.Sequential(
            nn.Linear(FUSION_DIM, 64),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(64, 1),
        )

    def pair_encoder_regularization_loss(self) -> torch.Tensor:
        """Wave9-compatible hook used by train_taskA (0 for MLP)."""
        total = None
        for enc in self.role_encoders:
            if hasattr(enc, "regularization_loss"):
                term = enc.regularization_loss()
                total = term if total is None else total + term
        if total is None:
            return next(self.parameters()).new_zeros(())
        return total

    def encode_g_stat(
        self,
        stat_raw: torch.Tensor,
        stat_role_masks: torch.Tensor,
    ) -> torch.Tensor:
        """stat_raw [B,3,D], masks [B,3] → g_stat [B,24]."""
        if self.force_zero_stat:
            return stat_raw.new_zeros((stat_raw.size(0), STAT_DIM))
        if stat_raw.ndim != 3 or stat_raw.size(1) != N_ROLES:
            raise ValueError(f"stat_raw must be [B,3,D], got {tuple(stat_raw.shape)}")
        if stat_raw.size(-1) != self.stat_raw_dim:
            raise ValueError(
                f"stat_raw D={stat_raw.size(-1)} != decoder D={self.stat_raw_dim}"
            )
        masks = stat_role_masks.to(device=stat_raw.device, dtype=stat_raw.dtype)
        if masks.ndim != 2 or masks.size(1) != N_ROLES:
            raise ValueError(
                f"stat_role_masks must be [B,3], got {tuple(masks.shape)}"
            )
        parts = []
        for r in range(N_ROLES):
            lat = self.role_encoders[r](stat_raw[:, r, :])
            parts.append(lat * masks[:, r].unsqueeze(-1))
        return torch.cat(parts, dim=1)

    def forward(
        self,
        embeddings: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        g_stat: torch.Tensor | None = None,
        stat_raw: torch.Tensor | None = None,
        stat_role_masks: torch.Tensor | None = None,
        **_kwargs,
    ) -> torch.Tensor:
        source = source.to(device=embeddings.device, dtype=torch.long).view(-1)
        target = target.to(device=embeddings.device, dtype=torch.long).view(-1)
        if source.numel() == 0:
            return embeddings.new_empty((0,))

        z_a = embeddings[source]
        z_g = embeddings[target]
        q = torch.cat([z_a, z_g, z_a * z_g, torch.abs(z_a - z_g)], dim=1)
        g_graph = self.graph_branch(q)
        batch = g_graph.size(0)

        if self.force_zero_stat or (
            g_stat is None and (stat_raw is None or stat_role_masks is None)
        ):
            g_stat_use = g_graph.new_zeros((batch, STAT_DIM))
        elif g_stat is not None and stat_raw is None:
            g_stat_use = g_stat.to(device=g_graph.device, dtype=g_graph.dtype)
        else:
            assert stat_raw is not None and stat_role_masks is not None
            g_stat_use = self.encode_g_stat(
                stat_raw.to(device=g_graph.device, dtype=g_graph.dtype),
                stat_role_masks,
            )

        if g_stat_use.ndim != 2 or g_stat_use.size(-1) != STAT_DIM:
            raise ValueError(
                f"g_stat must be [B,{STAT_DIM}], got {tuple(g_stat_use.shape)}"
            )
        if g_stat_use.size(0) != batch:
            raise ValueError(
                f"g_stat batch {g_stat_use.size(0)} != graph batch {batch}"
            )

        fused = torch.cat([g_graph, g_stat_use], dim=1)
        if fused.size(-1) != FUSION_DIM:
            raise RuntimeError(f"fusion dim {fused.size(-1)} != {FUSION_DIM}")
        return self.output_head(fused).squeeze(1)


def build_fusion88_stat_decoder(cfg: Any, hidden_channels: int) -> nn.Module:
    decoder_cfg = getattr(cfg.model, "decoder", None)
    dropout = float(
        getattr(decoder_cfg, "dropout", 0.2) if decoder_cfg is not None else 0.2
    )
    pair_dropout = float(
        getattr(decoder_cfg, "pair_dropout", PAIR_DROPOUT)
        if decoder_cfg is not None
        else PAIR_DROPOUT
    )
    force_zero = bool(
        getattr(decoder_cfg, "force_zero_stat", False)
        if decoder_cfg is not None
        else False
    )
    # Also treat S0 via experiment.stat_variant
    variant = str(
        getattr(getattr(cfg, "experiment", None), "stat_variant", "") or ""
    ).upper()
    if variant.startswith("S0") or variant in {"NONE", "S0_NONE"}:
        force_zero = True
    raw_dim = int(
        getattr(decoder_cfg, "stat_raw_dim", 1) if decoder_cfg is not None else 1
    )
    enc_kind = str(
        getattr(decoder_cfg, "stat_pair_encoder", "mlp")
        if decoder_cfg is not None
        else "mlp"
    )
    spline_l1 = float(
        getattr(decoder_cfg, "spline_l1", KAN_SPLINE_L1)
        if decoder_cfg is not None
        else KAN_SPLINE_L1
    )
    return Fusion88StatDecoder(
        hidden_channels=int(hidden_channels),
        stat_raw_dim=max(raw_dim, 1),
        dropout=dropout,
        pair_dropout=pair_dropout,
        force_zero_stat=force_zero,
        stat_pair_encoder=enc_kind,
        spline_l1=spline_l1,
    )


def build_stage_c_decoder(cfg: Any, hidden_channels: int) -> nn.Module:
    """Return Fusion88StatDecoder or plain Fusion88Decoder for S0 zeros path."""
    decoder_cfg = getattr(cfg.model, "decoder", None)
    force_zero = bool(
        getattr(decoder_cfg, "force_zero_stat", False)
        if decoder_cfg is not None
        else False
    )
    variant = str(
        getattr(getattr(cfg, "experiment", None), "stat_variant", "") or ""
    ).upper()
    if force_zero or variant.startswith("S0") or variant in {"NONE", "S0_NONE"}:
        # Exact Stage A decoder class for bit-comparable zeros path.
        return Fusion88Decoder(hidden_channels=int(hidden_channels), dropout=float(
            getattr(decoder_cfg, "dropout", 0.2) if decoder_cfg is not None else 0.2
        ))
    return build_fusion88_stat_decoder(cfg, hidden_channels)
