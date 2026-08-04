"""Leave-one-candidate-out path completion support (no candidate self-evidence)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import networkx as nx

from wnerw.finite_path_ensemble import FinitePathEnsemble, logsumexp
from wnerw.types import NEG_INF


def _ensure_dag(graph: nx.DiGraph) -> nx.DiGraph:
    g = graph.copy()
    if nx.is_directed_acyclic_graph(g):
        return g
    while True:
        try:
            cycle = nx.find_cycle(g, orientation="original")
        except nx.exception.NetworkXNoCycle:
            break
        u, v, *_ = cycle[0]
        g.remove_edge(u, v)
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("Expected a DAG after cycle peel.")
    return g


def graph_without_candidate(
    graph: nx.DiGraph,
    candidate_edge: tuple[str, str],
) -> nx.DiGraph:
    g = graph.copy()
    u, v = str(candidate_edge[0]), str(candidate_edge[1])
    if g.has_edge(u, v):
        g.remove_edge(u, v)
    return _ensure_dag(g)


def _filled_scores(
    graph: nx.DiGraph,
    edge_scores: Mapping[tuple[str, str], float],
) -> dict[tuple[str, str], float]:
    scores = {(str(a), str(b)): float(w) for (a, b), w in edge_scores.items()}
    for a, b in graph.edges:
        key = (str(a), str(b))
        if key not in scores:
            scores[key] = 0.0
    return scores


def multi_source_forward(
    graph: nx.DiGraph,
    edge_scores: Mapping[tuple[str, str], float],
    source_nodes: Sequence[str],
) -> dict[str, float]:
    """log F(u) = logsumexp over active sources of prefix mass s ↝ u."""
    g = _ensure_dag(graph)
    scores = _filled_scores(g, edge_scores)
    order = list(nx.topological_sort(g))
    log_f = {str(n): NEG_INF for n in g.nodes}
    for s in source_nodes:
        s = str(s)
        if s in log_f:
            log_f[s] = 0.0

    for node in order:
        node = str(node)
        mass = log_f[node]
        if mass == NEG_INF:
            continue
        for tgt in g.successors(node):
            tgt = str(tgt)
            cand = mass + scores[(node, tgt)]
            log_f[tgt] = logsumexp([log_f[tgt], cand])
    return log_f


def multi_endpoint_backward(
    graph: nx.DiGraph,
    edge_scores: Mapping[tuple[str, str], float],
    endpoint_nodes: Sequence[str],
) -> dict[str, float]:
    """log B(v) = logsumexp over endpoints of continuation mass v ↝ y."""
    g = _ensure_dag(graph)
    scores = _filled_scores(g, edge_scores)
    order = list(nx.topological_sort(g))
    endpoints = {str(y) for y in endpoint_nodes}
    log_b = {str(n): NEG_INF for n in g.nodes}
    for y in endpoints:
        if y in log_b:
            log_b[y] = 0.0

    for node in reversed(order):
        node = str(node)
        values: list[float] = []
        for tgt in g.successors(node):
            tgt = str(tgt)
            if log_b[tgt] == NEG_INF:
                continue
            values.append(scores[(node, tgt)] + log_b[tgt])
        if not values:
            continue
        merged = logsumexp(values)
        if log_b[node] == NEG_INF:
            log_b[node] = merged
        else:
            log_b[node] = logsumexp([log_b[node], merged])
    return log_b


def completion_from_tables(
    candidate_edge: tuple[str, str],
    log_forward: Mapping[str, float],
    log_backward: Mapping[str, float],
    source_nodes: Sequence[str],
    endpoint_nodes: Sequence[str],
) -> float:
    u, v = str(candidate_edge[0]), str(candidate_edge[1])
    sources = {str(s) for s in source_nodes}
    endpoints = {str(y) for y in endpoint_nodes}

    forward_terms: list[float] = []
    if u in sources:
        forward_terms.append(0.0)
    fu = float(log_forward.get(u, NEG_INF))
    if fu != NEG_INF and u not in sources:
        forward_terms.append(fu)

    backward_terms: list[float] = []
    if v in endpoints:
        backward_terms.append(0.0)
    bv = float(log_backward.get(v, NEG_INF))
    if bv != NEG_INF and v not in endpoints:
        backward_terms.append(bv)

    if not forward_terms or not backward_terms:
        return NEG_INF
    return float(logsumexp(forward_terms) + logsumexp(backward_terms))


def precompute_patient_support_tables(
    graph: nx.DiGraph,
    edge_scores: Mapping[tuple[str, str], float],
    source_nodes: Sequence[str],
    endpoint_nodes: Sequence[str],
) -> tuple[dict[str, float], dict[str, float]]:
    g = _ensure_dag(graph)
    return (
        multi_source_forward(g, edge_scores, source_nodes),
        multi_endpoint_backward(g, edge_scores, endpoint_nodes),
    )


def candidate_path_completion_score(
    graph: nx.DiGraph,
    source_nodes: list[str],
    endpoint_nodes: list[str],
    candidate_edge: tuple[str, str],
    patient_edge_scores: dict[tuple[str, str], float],
) -> float:
    """Path-completion support without using the candidate's own score/label."""
    g = graph_without_candidate(graph, candidate_edge)
    sources = [str(s) for s in source_nodes if str(s) in g]
    endpoints = [str(y) for y in endpoint_nodes if str(y) in g]
    if not sources or not endpoints:
        return NEG_INF
    log_f, log_b = precompute_patient_support_tables(
        g, patient_edge_scores, sources, endpoints
    )
    return completion_from_tables(
        candidate_edge, log_f, log_b, sources, endpoints
    )


def exact_log_partition_between(
    graph: nx.DiGraph,
    edge_scores: Mapping[tuple[str, str], float],
    source: str,
    endpoint: str,
) -> float:
    scores = _filled_scores(graph, edge_scores)
    ens = FinitePathEnsemble(graph, scores, temperature=1.0)
    return float(ens.log_partition_value(str(source), str(endpoint)))
