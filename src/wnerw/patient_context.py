"""Patient super-source helpers for conditioned WNERW (Stage 5 prep)."""

from __future__ import annotations

import math

import networkx as nx


def add_patient_super_source(
    graph: nx.DiGraph,
    active_sources: list[str],
    root_name: str = "patient_root",
) -> nx.DiGraph:
    """Attach a virtual root → each active exposure with uniform prior later."""
    result = graph.copy()
    result.add_node(root_name)
    for source in active_sources:
        source = str(source)
        if source not in result:
            result.add_node(source)
        result.add_edge(root_name, source)
    return result


def uniform_source_log_priors(active_sources: list[str]) -> dict[tuple[str, str], float]:
    """``log π(d|p) = -log |S_p|`` on edges ``patient_root → d``."""
    if not active_sources:
        return {}
    log_prior = -math.log(len(active_sources))
    return {("patient_root", str(s)): float(log_prior) for s in active_sources}
