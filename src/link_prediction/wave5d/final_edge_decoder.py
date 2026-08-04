"""Final patient-conditioned edge decoder + variant feature masks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class VariantFeatureMask:
    """Which blocks are active for a Wave 5D ablation variant."""

    use_hgt_pair: bool = True
    use_hcr: bool = True
    use_path_completion: bool = True
    use_gate: bool = True
    use_support: bool = True
    use_uncertainty: bool = True
    hcr_dim: int = 24  # 8 for binary_compact, 24 for structural pairwise


VARIANT_MASKS: dict[str, VariantFeatureMask] = {
    "L0_hgt": VariantFeatureMask(
        use_hgt_pair=True,
        use_hcr=False,
        use_path_completion=False,
        use_gate=False,
    ),
    "L1_pairwise_hcr": VariantFeatureMask(
        use_hgt_pair=True,
        use_hcr=True,
        use_path_completion=False,
        use_gate=False,
        hcr_dim=8,
    ),
    "L2_structural_hcr": VariantFeatureMask(
        use_hgt_pair=True,
        use_hcr=True,
        use_path_completion=False,
        use_gate=False,
        hcr_dim=24,
    ),
    "L3_path_support": VariantFeatureMask(
        use_hgt_pair=True,
        use_hcr=True,
        use_path_completion=True,
        use_gate=False,
        hcr_dim=24,
    ),
    "L4_final": VariantFeatureMask(
        use_hgt_pair=True,
        use_hcr=True,
        use_path_completion=True,
        use_gate=True,
        hcr_dim=24,
    ),
    "L5_patient_shuffle": VariantFeatureMask(
        use_hgt_pair=True,
        use_hcr=True,
        use_path_completion=True,
        use_gate=True,
        hcr_dim=24,
    ),
    "L6_context_shuffle": VariantFeatureMask(
        use_hgt_pair=True,
        use_hcr=True,
        use_path_completion=True,
        use_gate=True,
        hcr_dim=24,
    ),
    "L7_matched_random_context": VariantFeatureMask(
        use_hgt_pair=True,
        use_hcr=True,
        use_path_completion=True,
        use_gate=True,
        hcr_dim=24,
    ),
    "L8_random_path_weights": VariantFeatureMask(
        use_hgt_pair=True,
        use_hcr=True,
        use_path_completion=True,
        use_gate=True,
        hcr_dim=24,
    ),
    "L9_all_context_warning": VariantFeatureMask(
        use_hgt_pair=True,
        use_hcr=True,
        use_path_completion=True,
        use_gate=True,
        hcr_dim=24,
    ),
    "L10_no_leave_one_out": VariantFeatureMask(
        use_hgt_pair=True,
        use_hcr=True,
        use_path_completion=True,
        use_gate=True,
        hcr_dim=24,
    ),
}


def feature_dim(hgt_pair_dim: int, mask: VariantFeatureMask) -> int:
    dim = 0
    if mask.use_hgt_pair:
        dim += hgt_pair_dim
    if mask.use_hcr:
        dim += int(mask.hcr_dim)
    if mask.use_path_completion:
        dim += 1
    if mask.use_gate:
        dim += 1
    if mask.use_support:
        dim += 1
    if mask.use_uncertainty:
        dim += 1
    return dim


class FinalEdgeDecoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        if in_dim < 1:
            raise ValueError("in_dim must be positive")
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: [..., in_dim] → logits [...]"""
        return self.net(features).squeeze(-1)


def apply_feature_mask(
    *,
    hgt_pair: torch.Tensor,
    hcr24: torch.Tensor,
    path_completion: torch.Tensor,
    gate: torch.Tensor,
    support: torch.Tensor,
    uncertainty: torch.Tensor,
    mask: VariantFeatureMask,
) -> torch.Tensor:
    """Stack active feature blocks. Tensors broadcast on leading dims."""
    parts: list[torch.Tensor] = []
    if mask.use_hgt_pair:
        parts.append(hgt_pair)
    if mask.use_hcr:
        parts.append(hcr24[..., : mask.hcr_dim])
    if mask.use_path_completion:
        parts.append(path_completion.unsqueeze(-1))
    if mask.use_gate:
        parts.append(gate.unsqueeze(-1))
    if mask.use_support:
        parts.append(support.unsqueeze(-1))
    if mask.use_uncertainty:
        parts.append(uncertainty.unsqueeze(-1))
    return torch.cat(parts, dim=-1)
