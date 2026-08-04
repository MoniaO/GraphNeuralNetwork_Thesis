"""Feature-set registry for HCR decoder variants."""

from __future__ import annotations

from typing import Sequence

from .binary_features import (
    CLASSICAL_FEATURE_NAMES,
    COMPACT_FEATURE_NAMES,
    MINIMAL_FEATURE_NAMES,
)

VARIANT_FEATURES: dict[str, tuple[str, ...]] = {
    "none": tuple(),
    "binary_compact": COMPACT_FEATURE_NAMES,
    "binary_minimal": MINIMAL_FEATURE_NAMES,
    "classical_binary": CLASSICAL_FEATURE_NAMES,
    # Placebo uses compact width but zeros content at fit time.
    "all_zero": COMPACT_FEATURE_NAMES,
}

# HCR-3 variants use triple_features.HCR3_FEATURE_NAMES (handled outside).
HCR3_VARIANTS = frozenset(
    {"hcr3_full", "hcr3_without_a111", "hcr3_shuffled", "hcr3_random_context"}
)



def feature_names_for_variant(variant: str) -> Sequence[str]:
    key = str(variant).strip().lower()
    if key not in VARIANT_FEATURES:
        raise KeyError(
            f"Unknown HCR variant {variant!r}. "
            f"Available: {sorted(VARIANT_FEATURES)}"
        )
    return VARIANT_FEATURES[key]
