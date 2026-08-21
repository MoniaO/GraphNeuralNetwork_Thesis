"""Patient binary activity and AND-gate activation."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from copy import deepcopy
from math import prod

import pandas as pd


def build_patient_activity(
    patient_row: pd.Series,
    allowed_nodes: Collection[str],
    *,
    binary_threshold: float = 0.5,
) -> dict[str, float]:
    activity: dict[str, float] = {}
    for node in allowed_nodes:
        if node not in patient_row.index:
            activity[str(node)] = 0.0
            continue
        value = patient_row[node]
        if pd.isna(value):
            activity[str(node)] = 0.0
        else:
            activity[str(node)] = float(float(value) >= float(binary_threshold))
    return activity


def gate_activation(
    source: str,
    context_nodes: tuple[str, ...],
    patient_activity: Mapping[str, float],
) -> float:
    """AND potential: x_A * Π_B x_B."""
    source_active = float(patient_activity.get(str(source), 0.0))
    if not context_nodes:
        return source_active
    context_active = prod(
        float(patient_activity.get(str(node), 0.0)) for node in context_nodes
    )
    return float(source_active * context_active)


def make_gate_toggle_profiles(
    activity: dict[str, float],
    source: str,
    context_nodes: tuple[str, ...],
) -> tuple[dict[str, float], dict[str, float]]:
    """Controlled ON/OFF copies; remaining features identical."""
    profile_on = deepcopy(activity)
    profile_off = deepcopy(activity)
    profile_on[str(source)] = 1.0
    profile_off[str(source)] = 1.0
    for context in context_nodes:
        profile_on[str(context)] = 1.0
    if context_nodes:
        profile_off[str(context_nodes[0])] = 0.0
    return profile_on, profile_off


def make_triple_off_profiles(
    activity: dict[str, float],
    source: str,
    context_nodes: tuple[str, ...],
) -> list[dict[str, float]]:
    """OFF variants: each context zeroed once (triple-gate protocol)."""
    outs: list[dict[str, float]] = []
    for i, _ in enumerate(context_nodes):
        off = deepcopy(activity)
        off[str(source)] = 1.0
        for j, ctx in enumerate(context_nodes):
            off[str(ctx)] = 0.0 if j == i else 1.0
        # Ensure other contexts that should be ON for "missing one" are 1
        for j, ctx in enumerate(context_nodes):
            if j != i:
                off[str(ctx)] = 1.0
        outs.append(off)
    if not outs:
        off = deepcopy(activity)
        off[str(source)] = 1.0
        outs.append(off)
    return outs
