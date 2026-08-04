"""Top-K highest-weight finite paths via DAG dynamic programming."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .finite_path_ensemble import FinitePathEnsemble
from .types import NEG_INF


@dataclass(frozen=True)
class RankedPath:
    nodes: tuple[str, ...]
    log_weight: float
    probability: float


def top_k_paths(
    ensemble: FinitePathEnsemble,
    source: str,
    endpoint: str,
    k: int,
    log_partition: dict[str, float] | None = None,
) -> list[RankedPath]:
    """Return up to ``k`` highest-scoring paths from source to endpoint."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    source = str(source)
    endpoint = str(endpoint)

    if log_partition is None:
        log_partition = ensemble.backward_log_partition(endpoint)

    source_partition = log_partition[source]
    if source_partition == NEG_INF:
        return []

    best: dict[str, list[tuple[float, tuple[str, ...]]]] = {
        node: [] for node in ensemble.graph.nodes
    }
    best[endpoint] = [(0.0, (endpoint,))]

    for node in reversed(ensemble.topological_order):
        if node == endpoint:
            continue

        candidates: list[tuple[float, tuple[str, ...]]] = []
        for successor in ensemble.graph.successors(node):
            edge_weight = ensemble.edge_log_weights[(node, successor)]
            for tail_score, tail_path in best[successor]:
                candidates.append(
                    (edge_weight + tail_score, (node,) + tail_path)
                )

        candidates.sort(key=lambda item: item[0], reverse=True)
        best[node] = candidates[:k]

    return [
        RankedPath(
            nodes=path,
            log_weight=score,
            probability=float(np.exp(score - source_partition)),
        )
        for score, path in best[source]
    ]
