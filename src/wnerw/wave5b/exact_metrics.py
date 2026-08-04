"""Exact finite-path metrics for Wave 5B (TPM, entropy, hub mass)."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np

NEGATIVE_INFINITY = float("-inf")


def logsumexp(values: list[float]) -> float:
    if not values:
        return NEGATIVE_INFINITY
    maximum = max(values)
    if maximum == NEGATIVE_INFINITY:
        return NEGATIVE_INFINITY
    return float(maximum + np.log(np.exp(np.asarray(values) - maximum).sum()))


def backward_log_partition(
    graph: nx.DiGraph,
    edge_scores: dict[tuple[str, str], float],
    endpoint: str,
) -> dict[str, float]:
    order = list(nx.topological_sort(graph))
    backward = {str(node): NEGATIVE_INFINITY for node in graph.nodes}
    backward[str(endpoint)] = 0.0

    for source in reversed(order):
        source = str(source)
        if source == endpoint:
            continue
        continuations: list[float] = []
        for target in graph.successors(source):
            target = str(target)
            if backward[target] == NEGATIVE_INFINITY:
                continue
            continuations.append(
                edge_scores[(source, target)] + backward[target]
            )
        backward[source] = logsumexp(continuations)
    return backward


def path_log_score(
    path: list[str],
    edge_scores: dict[tuple[str, str], float],
) -> float:
    return float(
        sum(
            edge_scores[(str(u), str(v))]
            for u, v in zip(path[:-1], path[1:])
        )
    )


def true_path_mass(
    true_path: list[str],
    graph: nx.DiGraph,
    edge_scores: dict[tuple[str, str], float],
    backward: dict[str, float],
) -> float:
    if len(true_path) < 2:
        return 0.0
    source = str(true_path[0])
    has_all_edges = all(
        graph.has_edge(str(u), str(v))
        for u, v in zip(true_path[:-1], true_path[1:])
    )
    if not has_all_edges:
        return 0.0
    if backward.get(source, NEGATIVE_INFINITY) == NEGATIVE_INFINITY:
        return 0.0
    log_mass = path_log_score(true_path, edge_scores) - backward[source]
    return float(math.exp(log_mass))


def transition_probabilities(
    graph: nx.DiGraph,
    edge_scores: dict[tuple[str, str], float],
    backward: dict[str, float],
    source: str,
) -> dict[str, float]:
    source = str(source)
    source_partition = backward[source]
    if source_partition == NEGATIVE_INFINITY:
        return {}
    probabilities: dict[str, float] = {}
    for target in graph.successors(source):
        target = str(target)
        if backward[target] == NEGATIVE_INFINITY:
            continue
        log_probability = (
            edge_scores[(source, target)]
            + backward[target]
            - source_partition
        )
        probabilities[target] = float(math.exp(log_probability))
    return probabilities


def exact_path_entropy(
    graph: nx.DiGraph,
    edge_scores: dict[tuple[str, str], float],
    backward: dict[str, float],
    source: str,
    endpoint: str,
) -> float:
    order = list(nx.topological_sort(graph))
    entropy = {str(node): 0.0 for node in graph.nodes}
    entropy[str(endpoint)] = 0.0

    for node in reversed(order):
        node = str(node)
        if node == endpoint:
            continue
        transitions = transition_probabilities(
            graph=graph,
            edge_scores=edge_scores,
            backward=backward,
            source=node,
        )
        if not transitions:
            entropy[node] = 0.0
            continue
        local_entropy = -sum(
            p * math.log(max(p, 1e-15)) for p in transitions.values()
        )
        future_entropy = sum(
            p * entropy[t] for t, p in transitions.items()
        )
        entropy[node] = local_entropy + future_entropy
    return float(entropy[str(source)])


def exact_hub_mass(
    graph: nx.DiGraph,
    edge_scores: dict[tuple[str, str], float],
    backward: dict[str, float],
    source: str,
    endpoint: str,
    hubs: set[str],
) -> float:
    """P(path source→endpoint visits ≥1 hub)."""
    hubs = {str(h) for h in hubs}
    order = list(nx.topological_sort(graph))
    hit_probability = {str(node): 0.0 for node in graph.nodes}
    hit_probability[str(endpoint)] = 1.0 if str(endpoint) in hubs else 0.0

    for node in reversed(order):
        node = str(node)
        if node == endpoint:
            continue
        if node in hubs:
            hit_probability[node] = 1.0
            continue
        transitions = transition_probabilities(
            graph=graph,
            edge_scores=edge_scores,
            backward=backward,
            source=node,
        )
        hit_probability[node] = sum(
            p * hit_probability[t] for t, p in transitions.items()
        )
    return float(hit_probability[str(source)])


def count_paths(graph: nx.DiGraph, source: str, endpoint: str) -> int:
    source, endpoint = str(source), str(endpoint)
    if source not in graph or endpoint not in graph:
        return 0
    counts = {str(n): 0 for n in graph.nodes}
    counts[endpoint] = 1
    for node in reversed(list(nx.topological_sort(graph))):
        node = str(node)
        if node == endpoint:
            continue
        counts[node] = sum(counts[str(t)] for t in graph.successors(node))
    return int(counts[source])
