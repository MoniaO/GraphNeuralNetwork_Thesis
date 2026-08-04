"""Residual fusion decoder: B2 pair encoder ± gated legacy ± optional triple CMI."""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn

from hcr.wave7.panel_b_residual.constants import (
    GATE_SLOTS,
    MOTIF_B2_DIM,
    MOTIF_CLASSICAL4_DIM,
    MOTIF_LATENT_WITH_TRIPLE,
    MOTIF_PAIR_LATENT_DIM,
    MOTIF_V0_DIM,
    N_ROLES,
    PAIR_B2_DIM,
    PAIR_CLASSICAL4_DIM,
    PAIR_LATENT_DIM,
    PAIR_V0_DIM,
    TRIPLE_DIM,
    TRIPLE_LATENT_DIM,
)

LegacyMode = Literal["none", "v0_8", "classical4"]


class ResidualFusionHCRDecoder(nn.Module):
    """B2 is the base; legacy may add a gated BB residual; R4 may add triple CMI."""

    def __init__(
        self,
        hidden_channels: int,
        *,
        legacy_mode: LegacyMode = "v0_8",
        use_triple: bool = False,
        dropout: float = 0.2,
        pair_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_channels < 1:
            raise ValueError(f"hidden_channels must be positive, got {hidden_channels}")

        self.hidden_channels = int(hidden_channels)
        self.legacy_mode: LegacyMode = legacy_mode
        self.use_legacy_residual = legacy_mode != "none"
        self.use_triple = bool(use_triple)
        self.hcr_dim = MOTIF_B2_DIM
        self.hgt_in = 4 * self.hidden_channels
        motif_out = MOTIF_LATENT_WITH_TRIPLE if self.use_triple else MOTIF_PAIR_LATENT_DIM

        self.b2_encoder = nn.Sequential(
            nn.Linear(PAIR_B2_DIM, 16),
            nn.GELU(),
            nn.LayerNorm(16),
            nn.Dropout(pair_dropout),
            nn.Linear(16, PAIR_LATENT_DIM),
            nn.LayerNorm(PAIR_LATENT_DIM),
        )

        legacy_in = PAIR_V0_DIM if legacy_mode == "v0_8" else PAIR_CLASSICAL4_DIM
        if self.use_legacy_residual:
            self.legacy_encoder = nn.Sequential(
                nn.Linear(legacy_in, 8),
                nn.GELU(),
                nn.LayerNorm(8),
                nn.Linear(8, PAIR_LATENT_DIM),
            )
            nn.init.zeros_(self.legacy_encoder[-1].weight)
            nn.init.zeros_(self.legacy_encoder[-1].bias)
            self.gate = nn.Sequential(
                nn.Linear(len(GATE_SLOTS), 8),
                nn.GELU(),
                nn.Linear(8, 1),
            )
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.constant_(self.gate[-1].bias, -2.0)
        else:
            self.legacy_encoder = nn.Identity()
            self.gate = nn.Identity()

        if self.use_triple:
            self.triple_encoder = nn.Sequential(
                nn.Linear(TRIPLE_DIM, 8),
                nn.GELU(),
                nn.LayerNorm(8),
                nn.Linear(8, TRIPLE_LATENT_DIM),
                nn.LayerNorm(TRIPLE_LATENT_DIM),
            )
        else:
            self.triple_encoder = None

        self.hgt_branch = nn.Sequential(
            nn.Linear(self.hgt_in, self.hgt_in),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hgt_in, 64),
            nn.GELU(),
            nn.LayerNorm(64),
        )
        self.output_head = nn.Sequential(
            nn.Linear(64 + motif_out, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self.last_gate_stats: dict[str, float] = {}
        self.last_alpha: torch.Tensor | None = None

    def legacy_and_gate_parameters(self):
        if not self.use_legacy_residual:
            return []
        return list(self.legacy_encoder.parameters()) + list(self.gate.parameters())

    def _gate_features(self, b2_flat: torch.Tensor) -> torch.Tensor:
        cols = [b2_flat[:, s] for s in GATE_SLOTS]
        return torch.stack(cols, dim=1)

    def encode_motif(
        self,
        b2_motif: torch.Tensor,
        legacy_motif: torch.Tensor | None,
        bb_mask: torch.Tensor | None,
        triple_features: torch.Tensor | None = None,
        *,
        force_disable_legacy: bool = False,
        force_disable_triple: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = b2_motif.size(0)
        if b2_motif.size(1) != MOTIF_B2_DIM:
            raise ValueError(f"B2 motif dim {b2_motif.size(1)} != {MOTIF_B2_DIM}")

        b2_blocks = b2_motif.reshape(batch, N_ROLES, PAIR_B2_DIM)
        b2_flat = b2_blocks.reshape(batch * N_ROLES, PAIR_B2_DIM)
        b2_latent = self.b2_encoder(b2_flat)

        use_legacy = (
            self.use_legacy_residual
            and not force_disable_legacy
            and legacy_motif is not None
            and bb_mask is not None
        )
        if use_legacy:
            expected = MOTIF_V0_DIM if self.legacy_mode == "v0_8" else MOTIF_CLASSICAL4_DIM
            pair_in = PAIR_V0_DIM if self.legacy_mode == "v0_8" else PAIR_CLASSICAL4_DIM
            if legacy_motif.size(1) != expected:
                raise ValueError(
                    f"legacy motif dim {legacy_motif.size(1)} != {expected} "
                    f"(mode={self.legacy_mode})"
                )
            if bb_mask.shape != (batch, N_ROLES):
                raise ValueError(f"bb_mask shape {tuple(bb_mask.shape)} != {(batch, N_ROLES)}")
            legacy_blocks = legacy_motif.reshape(batch, N_ROLES, pair_in)
            legacy_flat = legacy_blocks.reshape(batch * N_ROLES, pair_in)
            legacy_latent = self.legacy_encoder(legacy_flat)
            alpha = torch.sigmoid(self.gate(self._gate_features(b2_flat)))
            mask_flat = bb_mask.reshape(batch * N_ROLES, 1).to(dtype=alpha.dtype)
            alpha = alpha * mask_flat
            residual = alpha * legacy_latent * mask_flat
            pair_latent = b2_latent + residual
        else:
            alpha = b2_latent.new_zeros((batch * N_ROLES, 1))
            pair_latent = b2_latent

        pair_motif = pair_latent.reshape(batch, MOTIF_PAIR_LATENT_DIM)
        alpha_roles = alpha.reshape(batch, N_ROLES, 1)
        self.last_alpha = alpha_roles.detach()
        self._update_gate_stats(alpha_roles, bb_mask if use_legacy else None)

        if self.use_triple and not force_disable_triple and triple_features is not None:
            if triple_features.size(1) != TRIPLE_DIM:
                raise ValueError(f"triple dim {triple_features.size(1)} != {TRIPLE_DIM}")
            t_lat = self.triple_encoder(triple_features)
            motif_latent = torch.cat([pair_motif, t_lat], dim=1)
        else:
            if self.use_triple:
                # Keep head width: zeros when disabled at runtime.
                motif_latent = torch.cat(
                    [pair_motif, pair_motif.new_zeros((batch, TRIPLE_LATENT_DIM))], dim=1
                )
            else:
                motif_latent = pair_motif
        return motif_latent, alpha_roles

    def _update_gate_stats(
        self,
        alpha_roles: torch.Tensor,
        bb_mask: torch.Tensor | None,
    ) -> None:
        a = alpha_roles.detach().float().cpu().reshape(-1)
        if a.numel() == 0:
            self.last_gate_stats = {}
            return
        q = torch.quantile(a, torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90]))
        stats: dict[str, float] = {
            "gate/mean": float(a.mean()),
            "gate/median": float(q[2]),
            "gate/p10": float(q[0]),
            "gate/p25": float(q[1]),
            "gate/p75": float(q[3]),
            "gate/p90": float(q[4]),
        }
        for r, name in enumerate(("AZ", "AG", "ZG")):
            ar = alpha_roles[:, r, 0].detach().float().cpu()
            stats[f"gate/{name}_mean"] = float(ar.mean())
            stats[f"gate/{name}_median"] = float(ar.median())
        if bb_mask is not None:
            m = bb_mask.detach().float().cpu().reshape(-1) > 0.5
            if m.any():
                stats["gate/bb_mean"] = float(a[m].mean())
                stats["gate/bb_median"] = float(a[m].median())
            if (~m).any():
                stats["gate/nonbb_mean"] = float(a[~m].mean())
        self.last_gate_stats = stats

    def forward(
        self,
        embeddings: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        hcr_features: torch.Tensor,
        hcr_legacy_features: torch.Tensor | None = None,
        hcr_bb_mask: torch.Tensor | None = None,
        hcr_triple_features: torch.Tensor | None = None,
        *,
        force_disable_legacy: bool = False,
        force_disable_triple: bool = False,
    ) -> torch.Tensor:
        source = source.to(device=embeddings.device, dtype=torch.long).view(-1)
        target = target.to(device=embeddings.device, dtype=torch.long).view(-1)
        if source.numel() != target.numel():
            raise ValueError("source/target length mismatch")
        if source.numel() == 0:
            return embeddings.new_empty((0,))

        b2 = hcr_features.to(device=embeddings.device, dtype=embeddings.dtype)
        legacy = (
            None
            if hcr_legacy_features is None
            else hcr_legacy_features.to(device=embeddings.device, dtype=embeddings.dtype)
        )
        mask = (
            None
            if hcr_bb_mask is None
            else hcr_bb_mask.to(device=embeddings.device, dtype=embeddings.dtype)
        )
        triple = (
            None
            if hcr_triple_features is None
            else hcr_triple_features.to(device=embeddings.device, dtype=embeddings.dtype)
        )

        z_a = embeddings[source]
        z_g = embeddings[target]
        hgt_lat = self.hgt_branch(
            torch.cat([z_a, z_g, z_a * z_g, torch.abs(z_a - z_g)], dim=1)
        )
        motif_lat, _ = self.encode_motif(
            b2,
            legacy,
            mask,
            triple,
            force_disable_legacy=force_disable_legacy,
            force_disable_triple=force_disable_triple,
        )
        return self.output_head(torch.cat([hgt_lat, motif_lat], dim=1)).squeeze(1)


def _resolve_legacy_mode(variant: str, decoder_cfg: Any) -> LegacyMode:
    if decoder_cfg is not None and getattr(decoder_cfg, "legacy_mode", None):
        mode = str(decoder_cfg.legacy_mode).strip().lower()
        if mode in {"none", "v0_8", "classical4"}:
            return mode  # type: ignore[return-value]
    v = variant.upper()
    if "R3_" in v or "R4_" in v or "CLASSICAL4" in v:
        return "classical4"
    if "R2_" in v or "GATED_LEGACY" in v:
        return "v0_8"
    if decoder_cfg is not None and getattr(decoder_cfg, "use_legacy_residual", None) is False:
        return "none"
    if decoder_cfg is not None and getattr(decoder_cfg, "use_legacy_residual", None) is True:
        return "v0_8"
    return "none"


def build_residual_fusion_decoder(cfg: Any, hidden_channels: int) -> ResidualFusionHCRDecoder:
    decoder_cfg = getattr(cfg.model, "decoder", None)
    variant = str(
        getattr(cfg.experiment, "variant", None)
        or getattr(getattr(cfg, "hcr", None), "variant", "")
    )
    legacy_mode = _resolve_legacy_mode(variant, decoder_cfg)
    use_triple = "R4_" in variant.upper() or "CONDITIONAL_EDGE" in variant.upper()
    if decoder_cfg is not None and getattr(decoder_cfg, "use_triple", None) is not None:
        use_triple = bool(decoder_cfg.use_triple)
    dropout = float(getattr(decoder_cfg, "dropout", 0.2) if decoder_cfg is not None else 0.2)
    return ResidualFusionHCRDecoder(
        hidden_channels=hidden_channels,
        legacy_mode=legacy_mode,
        use_triple=use_triple,
        dropout=dropout,
    )
