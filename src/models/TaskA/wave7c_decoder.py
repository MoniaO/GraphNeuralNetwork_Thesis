"""Wave 7C/D / Wave 9 / Wave 10 decoder: B2 motif + optional AG KAN residual."""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn

from hcr.wave7.panel_b_residual.constants import (
    MOTIF_B2_DIM,
    MOTIF_PAIR_LATENT_DIM,
    N_ROLES,
    PAIR_B2_DIM,
    PAIR_LATENT_DIM,
    TRIPLE_DIM,
    TRIPLE_LATENT_DIM,
)
from models.TaskA.pair_encoders import PairEncoderConfig, build_pair_encoder
from models.TaskA.pair_encoders.ag_kan_residual import AGKANResidual, AGKANResidualConfig

ArchMode = Literal["shared_mlp", "unshared_mlp", "global_motif", "shared_linear"]
AblationMode = Literal[
    "none",
    "ag_only",
    "no_direct_ag",
    "mask_type_support_only",
    "explicit_role_masks",
]

_C3_KEEP = list(range(32, 40))
AG_ROLE = 1  # PAIR_ROLES = (AZ, AG, ZG)


def _mlp_pair_encoder(pair_dropout: float = 0.1) -> nn.Module:
    """Legacy helper kept for tests; identical to MLPPairEncoder stack."""
    return build_pair_encoder(
        PairEncoderConfig(type="mlp", dropout=float(pair_dropout))
    )


class Wave7CDecoder(nn.Module):
    """B2 motif decoder; Wave 9 swaps pair encoders; Wave 10 adds AG KAN residual."""

    def __init__(
        self,
        hidden_channels: int,
        *,
        arch: ArchMode = "shared_mlp",
        ablation: AblationMode = "none",
        use_triple: bool = False,
        dropout: float = 0.2,
        pair_dropout: float = 0.1,
        pair_encoder_cfg: PairEncoderConfig | None = None,
        ag_kan_residual_cfg: AGKANResidualConfig | None = None,
    ) -> None:
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.arch: ArchMode = arch
        self.ablation: AblationMode = ablation
        self.use_triple = bool(use_triple)
        self.hcr_dim = MOTIF_B2_DIM
        self.hgt_in = 4 * self.hidden_channels
        self.use_explicit_masks = ablation == "explicit_role_masks"
        motif_out = MOTIF_PAIR_LATENT_DIM + (3 if self.use_explicit_masks else 0)
        if self.use_triple:
            motif_out += TRIPLE_LATENT_DIM

        pe_cfg = pair_encoder_cfg or PairEncoderConfig(
            type="mlp", dropout=float(pair_dropout)
        )
        if pair_encoder_cfg is None:
            pe_cfg.dropout = float(pair_dropout)
        self.pair_encoder_cfg = pe_cfg
        self.pair_encoder_type = pe_cfg.type

        if arch == "shared_mlp":
            self.pair_encoder = build_pair_encoder(pe_cfg)
            self.role_encoders = None
            self.global_encoder = None
        elif arch == "unshared_mlp":
            self.pair_encoder = None
            self.role_encoders = nn.ModuleList(
                [build_pair_encoder(pe_cfg) for _ in range(N_ROLES)]
            )
            self.global_encoder = None
        elif arch == "global_motif":
            self.pair_encoder = None
            self.role_encoders = None
            self.global_encoder = nn.Sequential(
                nn.Linear(MOTIF_B2_DIM, 48),
                nn.GELU(),
                nn.LayerNorm(48),
                nn.Dropout(pair_dropout),
                nn.Linear(48, MOTIF_PAIR_LATENT_DIM),
                nn.LayerNorm(MOTIF_PAIR_LATENT_DIM),
            )
        elif arch == "shared_linear":
            self.pair_encoder = nn.Linear(PAIR_B2_DIM, PAIR_LATENT_DIM)
            self.role_encoders = None
            self.global_encoder = None
        else:
            raise ValueError(f"unknown arch {arch}")

        res_cfg = ag_kan_residual_cfg or AGKANResidualConfig(enabled=False)
        self.ag_kan_residual_cfg = res_cfg
        self.use_ag_kan_residual = bool(res_cfg.enabled)
        self.ag_kan_residual: AGKANResidual | None
        if self.use_ag_kan_residual:
            if arch != "unshared_mlp":
                raise ValueError("AG KAN residual requires arch=unshared_mlp (A1)")
            self.ag_kan_residual = AGKANResidual(res_cfg)
        else:
            self.ag_kan_residual = None

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
        self.last_residual_stats: dict[str, float] = {}

    def encode_pairs(
        self,
        b2_motif: torch.Tensor,
        role_masks: torch.Tensor | None = None,
        nonbinary_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = b2_motif.size(0)
        blocks = b2_motif.reshape(batch, N_ROLES, PAIR_B2_DIM)
        if self.arch == "global_motif":
            assert self.global_encoder is not None
            flat = self.global_encoder(b2_motif)
            return flat.reshape(batch, N_ROLES, PAIR_LATENT_DIM)
        if self.arch == "unshared_mlp":
            assert self.role_encoders is not None
            outs = [self.role_encoders[r](blocks[:, r, :]) for r in range(N_ROLES)]
            lat = torch.stack(outs, dim=1)
        else:
            assert self.pair_encoder is not None
            flat = blocks.reshape(batch * N_ROLES, PAIR_B2_DIM)
            lat = self.pair_encoder(flat).reshape(batch, N_ROLES, PAIR_LATENT_DIM)

        if self.ag_kan_residual is not None:
            ag_mlp = lat[:, AG_ROLE, :]
            ag_mask = None
            if role_masks is not None:
                ag_mask = role_masks[:, AG_ROLE]
            else:
                # Support mask from raw block magnitude.
                ag_mask = (blocks[:, AG_ROLE, :].abs().sum(dim=-1) > 0).to(
                    dtype=ag_mlp.dtype
                )
            gated, alpha, raw = self.ag_kan_residual(
                blocks[:, AG_ROLE, :],
                role_mask=ag_mask,
                nonbinary_mask=nonbinary_mask,
            )
            lat = lat.clone()
            lat[:, AG_ROLE, :] = ag_mlp + gated
            with torch.no_grad():
                mlp_n = float(ag_mlp.detach().norm(dim=-1).mean())
                raw_n = float(raw.detach().norm(dim=-1).mean())
                ratio = float(raw_n / max(mlp_n, 1e-8))
                kan_diag = self.ag_kan_residual.kan.diagnostics(blocks[:, AG_ROLE, :])
                self.last_residual_stats = {
                    "gate_alpha": float(alpha.detach()),
                    "kan_residual_norm": raw_n,
                    "mlp_latent_norm": mlp_n,
                    "ratio_residual_to_mlp": ratio,
                    "gated_residual_norm": float(gated.detach().norm(dim=-1).mean()),
                    "base_path_norm": float(
                        self.ag_kan_residual.last_stats.get("base_path_norm", float("nan"))
                    ),
                    "spline_path_norm": float(
                        self.ag_kan_residual.last_stats.get("spline_path_norm", float("nan"))
                    ),
                    "base_weight_norm": kan_diag.get("base_weight_norm", float("nan")),
                    "spline_coefficient_norm": kan_diag.get(
                        "spline_coefficient_norm", float("nan")
                    ),
                    "fraction_outside_grid": kan_diag.get(
                        "input_out_of_grid_fraction", float("nan")
                    ),
                }
                self.last_gate_stats = {
                    f"gate/{k}": v for k, v in self.last_residual_stats.items()
                }
        return lat

    def pair_encoder_regularization_loss(self) -> torch.Tensor:
        """Spline L1 from KAN pair encoders and/or AG residual; 0 for pure MLP."""
        mods: list[nn.Module] = []
        if self.pair_encoder is not None:
            mods.append(self.pair_encoder)
        if self.role_encoders is not None:
            mods.extend(list(self.role_encoders))
        if self.ag_kan_residual is not None:
            mods.append(self.ag_kan_residual)
        total = None
        for m in mods:
            if hasattr(m, "regularization_loss"):
                term = m.regularization_loss()  # type: ignore[operator]
                total = term if total is None else total + term
        if total is None:
            return torch.zeros(())
        return total

    def ag_kan_trainable_parameters(self):
        if self.ag_kan_residual is None:
            return []
        return list(self.ag_kan_residual.parameters())

    def ag_mlp_parameters(self):
        """AG role MLP encoder params (role index 1)."""
        if self.role_encoders is None:
            return []
        return list(self.role_encoders[AG_ROLE].parameters())

    def decoder_head_parameters(self):
        params = list(self.hgt_branch.parameters()) + list(self.output_head.parameters())
        if self.triple_encoder is not None:
            params += list(self.triple_encoder.parameters())
        return params

    def apply_ablation(
        self,
        b2_motif: torch.Tensor,
        role_latents: torch.Tensor,
        role_masks: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch = b2_motif.size(0)
        lat = role_latents
        masks = role_masks

        if self.ablation == "ag_only":
            lat = lat.clone()
            lat[:, 0, :] = 0.0
            lat[:, 2, :] = 0.0
        elif self.ablation == "no_direct_ag":
            lat = lat.clone()
            lat[:, 1, :] = 0.0
        elif self.ablation == "explicit_role_masks":
            if masks is None:
                raw = b2_motif.reshape(batch, N_ROLES, PAIR_B2_DIM)
                masks = (raw.abs().sum(dim=-1) > 0).to(dtype=lat.dtype)
            masks = masks.to(device=lat.device, dtype=lat.dtype)
            lat = lat * masks.unsqueeze(-1)
            motif = lat.reshape(batch, MOTIF_PAIR_LATENT_DIM)
            return motif, masks

        motif = lat.reshape(batch, MOTIF_PAIR_LATENT_DIM)
        return motif, None

    @staticmethod
    def mask_type_support_only(b2_motif: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(b2_motif)
        batch = b2_motif.size(0)
        blocks = b2_motif.reshape(batch, N_ROLES, PAIR_B2_DIM)
        kept = torch.zeros_like(blocks)
        for s in _C3_KEEP:
            kept[:, :, s] = blocks[:, :, s]
        return kept.reshape(batch, MOTIF_B2_DIM)

    def forward(
        self,
        embeddings: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        hcr_features: torch.Tensor,
        hcr_role_masks: torch.Tensor | None = None,
        hcr_triple_features: torch.Tensor | None = None,
        hcr_nonbinary_mask: torch.Tensor | None = None,
        **_kwargs,
    ) -> torch.Tensor:
        source = source.to(device=embeddings.device, dtype=torch.long).view(-1)
        target = target.to(device=embeddings.device, dtype=torch.long).view(-1)
        if source.numel() == 0:
            return embeddings.new_empty((0,))
        b2 = hcr_features.to(device=embeddings.device, dtype=embeddings.dtype)
        if self.ablation == "mask_type_support_only":
            b2 = self.mask_type_support_only(b2)

        z_a = embeddings[source]
        z_g = embeddings[target]
        hgt = self.hgt_branch(
            torch.cat([z_a, z_g, z_a * z_g, torch.abs(z_a - z_g)], dim=1)
        )
        role_masks = None
        if hcr_role_masks is not None:
            role_masks = hcr_role_masks.to(
                device=embeddings.device, dtype=embeddings.dtype
            )
        role_lat = self.encode_pairs(
            b2, role_masks=role_masks, nonbinary_mask=hcr_nonbinary_mask
        )
        motif, masks = self.apply_ablation(b2, role_lat, role_masks)
        if masks is not None:
            motif = torch.cat([motif, masks], dim=1)
        if self.use_triple:
            assert self.triple_encoder is not None
            batch = motif.size(0)
            if hcr_triple_features is None:
                triple = motif.new_zeros((batch, TRIPLE_DIM))
            else:
                triple = hcr_triple_features.to(
                    device=embeddings.device, dtype=embeddings.dtype
                )
            t_mask = triple[:, 5:6]
            t_lat = self.triple_encoder(triple) * t_mask
            motif = torch.cat([motif, t_lat], dim=1)
        return self.output_head(torch.cat([hgt, motif], dim=1)).squeeze(1)


def resolve_wave7c_arch(variant: str) -> ArchMode:
    v = variant.upper()
    if "W9_" in v or "WAVE9" in v or "W10_" in v or "WAVE10" in v:
        if "SHARED_LINEAR" in v:
            return "shared_linear"
        if "GLOBAL" in v:
            return "global_motif"
        return "unshared_mlp"
    if "A1_" in v or "UNSHARED" in v:
        return "unshared_mlp"
    if "A2_" in v or "GLOBAL_MOTIF" in v:
        return "global_motif"
    if "A3_" in v or "SHARED_LINEAR" in v:
        return "shared_linear"
    return "shared_mlp"


def resolve_wave7c_ablation(variant: str) -> AblationMode:
    v = variant.upper()
    if "C1_" in v or "AG_ONLY" in v:
        return "ag_only"
    if "C2_" in v or "NO_DIRECT_AG" in v:
        return "no_direct_ag"
    if "C3_" in v or "MASK_TYPE_SUPPORT" in v:
        return "mask_type_support_only"
    if "C6_" in v or "EXPLICIT_ROLE_MASKS" in v or "ROLE_MASKS" in v:
        return "explicit_role_masks"
    return "none"


def resolve_wave7c_use_triple(variant: str, decoder_cfg: Any) -> bool:
    v = variant.upper()
    # Wave 10 T1 is type-routed residual, NOT conditional-edge triple.
    if "W10_" in v or "WAVE10" in v:
        if decoder_cfg is not None and getattr(decoder_cfg, "use_triple", None) is not None:
            return bool(decoder_cfg.use_triple)
        return False
    if "T1_" in v or "CONDITIONAL_EDGE" in v or "PLUS_CONDITIONAL" in v:
        return True
    if decoder_cfg is not None and getattr(decoder_cfg, "use_triple", None) is not None:
        return bool(decoder_cfg.use_triple)
    return False


def resolve_pair_encoder_cfg(variant: str, decoder_cfg: Any, pair_dropout: float) -> PairEncoderConfig:
    raw = None
    if decoder_cfg is not None and getattr(decoder_cfg, "pair_encoder", None) is not None:
        raw = decoder_cfg.pair_encoder
    cfg = PairEncoderConfig.from_mapping(raw)
    v = variant.upper()
    explicit_type = None
    if raw is not None:
        explicit_type = getattr(raw, "type", None)
        if explicit_type is None and isinstance(raw, dict):
            explicit_type = raw.get("type")
    if not explicit_type:
        # Wave 10 residual keeps MLP base; only full replacement uses type=kan.
        if "W10_" in v or "WAVE10" in v:
            cfg.type = "mlp"
        elif "K1_" in v or "PAIR_KAN" in v or ("_KAN_" in v and "MLP" not in v and "RESIDUAL" not in v):
            cfg.type = "kan"
        else:
            cfg.type = "mlp"
    if explicit_type is None or getattr(raw, "dropout", None) is None:
        if not (isinstance(raw, dict) and "dropout" in raw):
            cfg.dropout = float(pair_dropout)
    return cfg


def resolve_ag_kan_residual_cfg(variant: str, decoder_cfg: Any) -> AGKANResidualConfig:
    raw = None
    if decoder_cfg is not None and getattr(decoder_cfg, "ag_kan_residual", None) is not None:
        raw = decoder_cfg.ag_kan_residual
    cfg = AGKANResidualConfig.from_mapping(raw)
    v = variant.upper()
    if raw is None or (hasattr(raw, "get") is False and getattr(raw, "enabled", None) is None):
        # Infer from variant when config block absent.
        if "W10_R1_" in v or "W10_R2_" in v or "AG_KAN_RESIDUAL" in v:
            cfg.enabled = True
        if "W10_T1_" in v or "TYPE_ROUTED" in v:
            cfg.enabled = True
            cfg.type_routed = True
        if "W10_G1_" in v or "ADAPTIVE_GRID" in v:
            cfg.enabled = True
            cfg.grid_update = True
        if "W10_R2_" in v or "FINETUNE" in v:
            cfg.train_mode = "finetune"
    # Explicit OmegaConf enabled=false wins.
    if raw is not None:
        enabled = getattr(raw, "enabled", None)
        if enabled is None and isinstance(raw, dict):
            enabled = raw.get("enabled")
        if enabled is not None:
            cfg.enabled = bool(enabled)
        tr = getattr(raw, "type_routed", None)
        if tr is None and isinstance(raw, dict):
            tr = raw.get("type_routed")
        if tr is not None:
            cfg.type_routed = bool(tr)
    return cfg


def build_wave7c_decoder(cfg: Any, hidden_channels: int) -> Wave7CDecoder:
    decoder_cfg = getattr(cfg.model, "decoder", None)
    variant = str(
        getattr(cfg.experiment, "variant", None)
        or getattr(getattr(cfg, "hcr", None), "variant", "")
    )
    arch = resolve_wave7c_arch(variant)
    ablation = resolve_wave7c_ablation(variant)
    use_triple = resolve_wave7c_use_triple(variant, decoder_cfg)
    dropout = float(getattr(decoder_cfg, "dropout", 0.2) if decoder_cfg is not None else 0.2)
    pair_dropout = float(
        getattr(decoder_cfg, "pair_dropout", 0.1) if decoder_cfg is not None else 0.1
    )
    if decoder_cfg is not None:
        if getattr(decoder_cfg, "arch", None):
            arch = str(decoder_cfg.arch)  # type: ignore[assignment]
        if getattr(decoder_cfg, "ablation", None):
            ablation = str(decoder_cfg.ablation)  # type: ignore[assignment]
    pe_cfg = resolve_pair_encoder_cfg(variant, decoder_cfg, pair_dropout)
    ag_cfg = resolve_ag_kan_residual_cfg(variant, decoder_cfg)
    return Wave7CDecoder(
        hidden_channels=hidden_channels,
        arch=arch,
        ablation=ablation,
        use_triple=use_triple,
        dropout=dropout,
        pair_dropout=pair_dropout,
        pair_encoder_cfg=pe_cfg,
        ag_kan_residual_cfg=ag_cfg,
    )
