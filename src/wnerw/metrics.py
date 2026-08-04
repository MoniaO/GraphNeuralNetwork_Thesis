"""Path-distribution diagnostics for Wave 5 (population + patient-conditioned)."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np

from .topk_paths import RankedPath


def path_entropy(probabilities: Sequence[float]) -> float:
    probs = [float(p) for p in probabilities if float(p) > 0.0]
    if not probs:
        return 0.0
    return float(-sum(p * math.log(p) for p in probs))


def path_hhi(probabilities: Sequence[float]) -> float:
    return float(sum(float(p) ** 2 for p in probabilities))


def true_path_mass(
    ranked: Sequence[RankedPath],
    true_paths: Iterable[tuple[str, ...]],
) -> float:
    """Sum of ranked probability mass on any path in ``true_paths``."""
    true_set = {tuple(p) for p in true_paths}
    return float(sum(p.probability for p in ranked if p.nodes in true_set))


def designated_true_path_mass(
    path_probability: float | None,
    *,
    has_true_path: bool,
) -> float:
    """Wave-5 TPM: P(γ*) if γ* ∈ Ω, else 0. Never drop the query."""
    if not has_true_path or path_probability is None:
        return 0.0
    return float(path_probability)


def path_recall_at_k(
    ranked: Sequence[RankedPath],
    true_paths: Iterable[tuple[str, ...]],
    k: int,
) -> float:
    true_set = {tuple(p) for p in true_paths}
    top = ranked[:k]
    if not true_set:
        return float("nan")
    hits = sum(1 for p in top if p.nodes in true_set)
    return float(hits / len(true_set))


def hit_at_k(
    ranked: Sequence[RankedPath],
    true_path: tuple[str, ...] | None,
    k: int,
) -> int:
    """1 if designated true path appears in top-k, else 0."""
    if true_path is None or not ranked:
        return 0
    target = tuple(true_path)
    return int(any(p.nodes == target for p in ranked[:k]))


def reciprocal_rank(
    ranked: Sequence[RankedPath],
    true_path: tuple[str, ...] | None,
) -> float:
    """1/rank of designated true path; 0 if absent."""
    if true_path is None:
        return 0.0
    target = tuple(true_path)
    for i, p in enumerate(ranked, start=1):
        if p.nodes == target:
            return 1.0 / float(i)
    return 0.0


def true_path_rank(
    ranked: Sequence[RankedPath],
    true_path: tuple[str, ...] | None,
) -> float:
    """1-based rank of designated true path; NaN if absent."""
    if true_path is None:
        return float("nan")
    target = tuple(true_path)
    for i, p in enumerate(ranked, start=1):
        if p.nodes == target:
            return float(i)
    return float("nan")


def hub_mass(
    ranked: Sequence[RankedPath],
    hub_nodes: Iterable[str],
) -> float:
    hubs = {str(h) for h in hub_nodes}
    return float(
        sum(p.probability for p in ranked if hubs.intersection(p.nodes))
    )


def mean_true_path_rank(
    ranked: Sequence[RankedPath],
    true_paths: Iterable[tuple[str, ...]],
) -> float:
    true_set = {tuple(p) for p in true_paths}
    ranks = [
        i + 1
        for i, p in enumerate(ranked)
        if p.nodes in true_set
    ]
    if not ranks:
        return float("nan")
    return float(np.mean(ranks))
