"""Frozen Wave-5 gate registry and soft/hard gate potentials."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from hcr.motif_registry_v3 import DUAL_GATES, GATE_OUTCOMES
from hcr.motifs import GATE_DEFINITIONS, OLD_GATES
from wnerw.potentials import safe_log_probability


@dataclass(frozen=True)
class GateSpec:
    gate_node: str
    required_inputs: tuple[str, ...]
    outcome_nodes: tuple[str, ...] = ()


def build_gate_registry() -> dict[str, GateSpec]:
    """Freeze generator gate definitions into Wave-5 GateSpec objects."""
    registry: dict[str, GateSpec] = {}
    merged = {**GATE_DEFINITIONS, **OLD_GATES, **DUAL_GATES}
    for gate, parents in merged.items():
        y = GATE_OUTCOMES.get(str(gate))
        registry[str(gate)] = GateSpec(
            gate_node=str(gate),
            required_inputs=tuple(str(p) for p in parents),
            outcome_nodes=(str(y),) if y else (),
        )
    return registry


GATES: dict[str, GateSpec] = build_gate_registry()


def gate_potential(
    gate: GateSpec,
    *,
    active_inputs: set[str] | None = None,
    motif_score: float = 0.0,
    reliability: float = 1.0,
    mode: str = "population",
    inactive_penalty: float = 5.0,
) -> float:
    """Return additive log-potential applied once when entering the gate node.

    Modes:
      population — ``motif_score * reliability`` (no patient context)
      hard       — ``-inf`` unless all required inputs are active
      soft       — subtract ``inactive_penalty`` when inputs missing
    """
    if mode == "population":
        return float(motif_score) * float(reliability)

    active = set(active_inputs or ())
    required_active = all(parent in active for parent in gate.required_inputs)
    if required_active:
        return float(motif_score) * float(reliability)
    if mode == "hard":
        return float("-inf")
    if mode == "soft":
        return -float(inactive_penalty)
    raise ValueError(f"Unknown gate mode: {mode!r}")


def apply_gate_potentials_to_edge_weights(
    edge_log_weights: dict[tuple[str, str], float],
    gate_scores: dict[str, float],
) -> dict[tuple[str, str], float]:
    """Add gate potential exactly once on edges whose *target* is a scored gate."""
    out = dict(edge_log_weights)
    for (source, target), weight in list(out.items()):
        if target in gate_scores:
            out[(source, target)] = weight + float(gate_scores[target])
    return out


def _edge_prob_lookup(
    evidence_rows: list[Mapping[str, Any]] | Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    probability_key: str = "p_d1_calibrated",
) -> dict[tuple[str, str], float]:
    if isinstance(evidence_rows, Mapping):
        items = evidence_rows.items()
        out: dict[tuple[str, str], float] = {}
        for key, row in items:
            p = row.get(probability_key, row.get("p_d1", row.get("p_calibrated")))
            if p is None:
                continue
            try:
                out[(str(key[0]), str(key[1]))] = float(p)
            except (TypeError, ValueError, IndexError):
                continue
        return out

    out = {}
    for row in evidence_rows:
        src, tgt = str(row["source"]), str(row["target"])
        p = row.get(probability_key, row.get("p_d1", row.get("p_calibrated")))
        if p is None:
            continue
        try:
            out[(src, tgt)] = float(p)
        except (TypeError, ValueError):
            continue
    return out


def population_gate_scores_from_d1(
    evidence_rows: list[Mapping[str, Any]] | Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    gates: Mapping[str, GateSpec] | None = None,
    coefficient: float = 1.0,
    probability_key: str = "p_d1_calibrated",
    min_parents: int = 1,
) -> dict[str, float]:
    """Population motif score = coefficient × mean log P_D1(parent→gate).

    Uses structural / registry parents. Missing parent→gate evidence is skipped.
    Gates with fewer than ``min_parents`` observed parent edges are omitted.
    """
    registry = gates or GATES
    probs = _edge_prob_lookup(evidence_rows, probability_key=probability_key)
    scores: dict[str, float] = {}
    for gate_name, spec in registry.items():
        logs: list[float] = []
        for parent in spec.required_inputs:
            p = probs.get((str(parent), str(gate_name)))
            if p is None:
                continue
            logs.append(safe_log_probability(p))
        if len(logs) < int(min_parents):
            continue
        scores[str(gate_name)] = float(coefficient) * (sum(logs) / len(logs))
    return scores
