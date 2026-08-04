"""Placeholders for binary–continuous / mixed HCR estimators (wave 3+)."""

from __future__ import annotations

from .variable_spec import VariableSpec


def encode_mixed_pair(source_spec: VariableSpec, target_spec: VariableSpec) -> None:
    raise NotImplementedError(
        "Mixed-type HCR pairs are not implemented in wave-3 stage 1: "
        f"{source_spec.variable_type} -> {target_spec.variable_type}"
    )
