"""Exact hidden-edge mass (HEM) via forward × transition probabilities."""

from __future__ import annotations

import networkx as nx

from wnerw.wave5b.exact_metrics import transition_probabilities


def forward_node_probabilities(
    graph: nx.DiGraph,
    edge_scores: dict[tuple[str, str], float],
    backward: dict[str, float],
    source: str,
) -> dict[str, float]:
    order = list(nx.topological_sort(graph))
    forward = {str(node): 0.0 for node in graph.nodes}
    forward[str(source)] = 1.0

    for node in order:
        node = str(node)
        if forward[node] == 0.0:
            continue
        transitions = transition_probabilities(
            graph=graph,
            edge_scores=edge_scores,
            backward=backward,
            source=node,
        )
        for target, probability in transitions.items():
            forward[str(target)] += forward[node] * float(probability)
    return forward


def exact_edge_mass(
    hidden_edge: tuple[str, str],
    graph: nx.DiGraph,
    edge_scores: dict[tuple[str, str], float],
    backward: dict[str, float],
    source: str,
) -> float:
    edge_source, edge_target = str(hidden_edge[0]), str(hidden_edge[1])
    if not graph.has_edge(edge_source, edge_target):
        return 0.0
    forward = forward_node_probabilities(
        graph=graph,
        edge_scores=edge_scores,
        backward=backward,
        source=str(source),
    )
    transitions = transition_probabilities(
        graph=graph,
        edge_scores=edge_scores,
        backward=backward,
        source=edge_source,
    )
    return float(forward.get(edge_source, 0.0) * transitions.get(edge_target, 0.0))


def enumerate_edge_mass(
    hidden_edge: tuple[str, str],
    graph: nx.DiGraph,
    edge_scores: dict[tuple[str, str], float],
    source: str,
    endpoint: str,
    *,
    cutoff: int = 12,
) -> float:
    """Brute-force HEM for unit tests (all simple paths)."""
    from wnerw.wave5b.exact_metrics import (
        backward_log_partition,
        path_log_score,
    )
    import math

    if not nx.has_path(graph, source, endpoint):
        return 0.0
    back = backward_log_partition(graph, edge_scores, endpoint)
    log_z = back[str(source)]
    if log_z == float("-inf"):
        return 0.0
    he = (str(hidden_edge[0]), str(hidden_edge[1]))
    mass = 0.0
    for path in nx.all_simple_paths(graph, source, endpoint, cutoff=cutoff):
        edges = list(zip(path[:-1], path[1:]))
        if he not in [(str(u), str(v)) for u, v in edges]:
            continue
        log_w = path_log_score([str(n) for n in path], edge_scores)
        mass += math.exp(log_w - log_z)
    return float(mass)
