"""Per-query evaluation for Wave 5B panels."""

from __future__ import annotations

from itertools import islice

import networkx as nx
import numpy as np

from .exact_metrics import (
    backward_log_partition,
    count_paths,
    exact_hub_mass,
    exact_path_entropy,
    true_path_mass,
)


def parse_true_path(text: object) -> list[str]:
    s = str(text or "").strip()
    if not s:
        return []
    return [p.strip() for p in s.split(">") if p.strip()]


def parse_hidden_edge(
    text: object,
    true_path: list[str],
) -> tuple[str, str] | None:
    s = str(text or "").strip()
    if s and "->" in s:
        a, b = s.split("->", 1)
        return str(a).strip(), str(b).strip()
    if s and ">" in s:
        parts = [p.strip() for p in s.split(">") if p.strip()]
        if len(parts) >= 2:
            return parts[0], parts[1]
    # Fallback: first hop of designated true path (population registry has empty hide).
    if len(true_path) >= 2:
        return true_path[0], true_path[1]
    return None


def top_k_paths(
    graph: nx.DiGraph,
    edge_scores: dict[tuple[str, str], float],
    source: str,
    endpoint: str,
    k: int,
) -> list[list[str]]:
    weighted = graph.copy()
    for u, v in weighted.edges:
        weighted[u][v]["cost"] = float(-edge_scores[(str(u), str(v))])
    if source not in weighted or endpoint not in weighted:
        return []
    if not nx.has_path(weighted, source, endpoint):
        return []
    generator = nx.shortest_simple_paths(
        weighted, source=source, target=endpoint, weight="cost"
    )
    return [[str(n) for n in path] for path in islice(generator, k)]


def evaluate_query(
    query_id: str,
    source: str,
    endpoint: str,
    true_path: list[str],
    hidden_edge: tuple[str, str] | None,
    graph: nx.DiGraph,
    edge_scores: dict[tuple[str, str], float],
    hubs: set[str],
    top_k: int = 100,
    *,
    compute_n_paths: bool = False,
) -> dict[str, object]:
    source, endpoint = str(source), str(endpoint)
    true_path = [str(n) for n in true_path]

    empty = {
        "query_id": query_id,
        "source": source,
        "endpoint": endpoint,
        "has_any_path": False,
        "has_true_path": False,
        "true_path_mass": 0.0,
        "true_path_rank_at_100": float("inf"),
        "reciprocal_rank_at_100": 0.0,
        "true_path_hit_at_1": 0,
        "true_path_hit_at_5": 0,
        "true_path_hit_at_10": 0,
        "hidden_edge_hit_at_1": 0,
        "hidden_edge_hit_at_5": 0,
        "hidden_edge_hit_at_10": 0,
        "path_entropy": 0.0,
        "hub_mass": 0.0,
        "n_paths": 0,
    }

    has_any_path = (
        source in graph
        and endpoint in graph
        and nx.has_path(graph, source, endpoint)
    )
    has_true_path = bool(true_path) and all(
        graph.has_edge(u, v) for u, v in zip(true_path[:-1], true_path[1:])
    )

    if not has_any_path:
        empty["has_true_path"] = bool(has_true_path)
        return empty

    backward = backward_log_partition(
        graph=graph, edge_scores=edge_scores, endpoint=endpoint
    )
    mass = true_path_mass(
        true_path=true_path,
        graph=graph,
        edge_scores=edge_scores,
        backward=backward,
    )
    paths = top_k_paths(
        graph=graph,
        edge_scores=edge_scores,
        source=source,
        endpoint=endpoint,
        k=top_k,
    )
    try:
        rank = float(paths.index(true_path) + 1)
    except ValueError:
        rank = float("inf")

    def true_path_hit(k: int) -> int:
        return int(any(path == true_path for path in paths[:k]))

    def hidden_edge_hit(k: int) -> int:
        if hidden_edge is None:
            return 0
        he = (str(hidden_edge[0]), str(hidden_edge[1]))
        return int(
            any(
                he in list(zip(path[:-1], path[1:]))
                for path in paths[:k]
            )
        )

    reciprocal_rank = 0.0 if not np.isfinite(rank) else 1.0 / float(rank)
    n_paths = count_paths(graph, source, endpoint) if compute_n_paths else 0

    return {
        "query_id": query_id,
        "source": source,
        "endpoint": endpoint,
        "has_any_path": True,
        "has_true_path": bool(has_true_path),
        "true_path_mass": float(mass if has_true_path else 0.0),
        "true_path_rank_at_100": rank,
        "reciprocal_rank_at_100": reciprocal_rank if has_true_path else 0.0,
        "true_path_hit_at_1": true_path_hit(1) if has_true_path else 0,
        "true_path_hit_at_5": true_path_hit(5) if has_true_path else 0,
        "true_path_hit_at_10": true_path_hit(10) if has_true_path else 0,
        "hidden_edge_hit_at_1": hidden_edge_hit(1),
        "hidden_edge_hit_at_5": hidden_edge_hit(5),
        "hidden_edge_hit_at_10": hidden_edge_hit(10),
        "path_entropy": exact_path_entropy(
            graph=graph,
            edge_scores=edge_scores,
            backward=backward,
            source=source,
            endpoint=endpoint,
        ),
        "hub_mass": exact_hub_mass(
            graph=graph,
            edge_scores=edge_scores,
            backward=backward,
            source=source,
            endpoint=endpoint,
            hubs=hubs,
        ),
        "n_paths": int(n_paths),
    }
