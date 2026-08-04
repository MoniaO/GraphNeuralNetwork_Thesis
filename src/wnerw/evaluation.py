"""Wave 5 evaluation helpers (G_true used only as evaluator oracle)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import networkx as nx
import pandas as pd

from .metrics import (
    hub_mass,
    mean_true_path_rank,
    path_entropy,
    path_hhi,
    path_recall_at_k,
    true_path_mass,
)
from .topk_paths import RankedPath


def true_paths_between(
    graph_true: nx.DiGraph,
    source: str,
    endpoint: str,
    *,
    max_paths: int = 200,
    cutoff: int | None = 8,
) -> list[tuple[str, ...]]:
    """Enumerate true directed paths source→endpoint (evaluator only)."""
    source, endpoint = str(source), str(endpoint)
    if source not in graph_true or endpoint not in graph_true:
        return []
    if not nx.has_path(graph_true, source, endpoint):
        return []
    out: list[tuple[str, ...]] = []
    for path in nx.all_simple_paths(
        graph_true, source, endpoint, cutoff=cutoff
    ):
        out.append(tuple(str(n) for n in path))
        if len(out) >= max_paths:
            break
    return out


def score_query_against_truth(
    ranked: Sequence[RankedPath],
    true_paths: Iterable[tuple[str, ...]],
    *,
    hub_nodes: Iterable[str] | None = None,
    ks: Sequence[int] = (1, 5, 10, 20),
) -> dict[str, float]:
    true_list = list(true_paths)
    probs = [p.probability for p in ranked]
    out: dict[str, float] = {
        "n_ranked": float(len(ranked)),
        "n_true_paths": float(len(true_list)),
        "path_entropy": path_entropy(probs),
        "path_hhi": path_hhi(probs),
        "true_path_mass": true_path_mass(ranked, true_list),
        "mean_true_path_rank": mean_true_path_rank(ranked, true_list),
    }
    for k in ks:
        out[f"path_recall_at_{k}"] = path_recall_at_k(ranked, true_list, k)
    if hub_nodes is not None:
        out["hub_mass"] = hub_mass(ranked, hub_nodes)
    return out


def truth_digraph_from_edges(edges: pd.DataFrame) -> nx.DiGraph:
    g = nx.DiGraph()
    for s, t in edges[["source", "target"]].itertuples(index=False, name=None):
        g.add_edge(str(s), str(t))
    return g
