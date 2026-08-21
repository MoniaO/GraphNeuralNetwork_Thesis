"""S0–S10 statistical variant registry (Stage C, 11.08).

What it does
------------
Single list of names and raw_dim values. FINAL uses S10_HCR_FULL40 (40D).
`get_variant("S10")` / `get_variant("S10_HCR_FULL40")` return the same entry.

What you may change
-------------------
Adding a new S11+ — only in a new experiment, new id.
Do not change raw_dim of existing S0–S10: the 11.08/14.08 tables would break.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StageCVariant:
    id: str
    name: str
    raw_dim: int
    kind: str  # none | classical | hcr_binary | hcr_slice
    description: str

    @property
    def config_id(self) -> str:
        return f"{self.id}_{self.name}"


STAGE_C_VARIANTS: tuple[StageCVariant, ...] = (
    StageCVariant("S0", "NONE", 0, "none", "No statistical evidence; g_stat=zeros(24)"),
    StageCVariant("S1", "NMI", 1, "classical", "Train-only normalized mutual information"),
    StageCVariant("S2", "JACCARD_ACTIVE", 1, "classical", "Jaccard on train-only activity sets"),
    StageCVariant("S3", "COSINE_ACTIVE", 1, "classical", "Cosine on train-only activity sets"),
    StageCVariant(
        "S4",
        "HCR_BINARY_ONLY",
        1,
        "hcr_binary",
        "Signed Pearson phi / a11; BB only (else role mask 0)",
    ),
    StageCVariant("S5", "META8", 8, "hcr_slice", "FULL40[32:40] support/types"),
    StageCVariant("S6", "HCR_MATRIX16", 16, "hcr_slice", "vec(padded 4x4 HCR)"),
    StageCVariant("S7", "HCR_COMPACT24", 24, "hcr_slice", "HCR16 || META8"),
    StageCVariant(
        "S8", "HCR_COMPACT32", 32, "hcr_slice", "HCR16 || marg_U4 || marg_V4 || META8"
    ),
    StageCVariant(
        "S9",
        "HCR_COMPACT36",
        36,
        "hcr_slice",
        "HCR16 || marg_U4 || marg_V4 || joint4 || META8",
    ),
    StageCVariant(
        "S10",
        "HCR_FULL40",
        40,
        "hcr_slice",
        "HCR16 || energy4 || marg_U4 || marg_V4 || joint4 || META8",
    ),
)

_BY_ID = {v.id: v for v in STAGE_C_VARIANTS}
_BY_CONFIG = {v.config_id: v for v in STAGE_C_VARIANTS}


def get_variant(name: str) -> StageCVariant:
    key = str(name).strip().upper()
    if key in _BY_ID:
        return _BY_ID[key]
    if key in _BY_CONFIG:
        return _BY_CONFIG[key]
    # Allow S1_NMI style where middle parts vary
    for v in STAGE_C_VARIANTS:
        if key == v.config_id or key.startswith(v.id + "_") or key == v.name:
            return v
    raise KeyError(f"Unknown Stage C variant: {name!r}")


def all_variant_ids() -> tuple[str, ...]:
    return tuple(v.id for v in STAGE_C_VARIANTS)
