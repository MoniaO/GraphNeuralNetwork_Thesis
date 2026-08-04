"""DP partition must match full path enumeration within 1e-8."""

from __future__ import annotations

import itertools
import math

import networkx as nx
import numpy as np

from wnerw.finite_path_ensemble import FinitePathEnsemble
from wnerw.path_marginals import edge_marginals, node_marginals
from wnerw.topk_paths import top_k_paths
from wnerw.types import NEG_INF


def _enumerate_paths(
    graph: nx.DiGraph,
    source: str,
    endpoint: str,
    edge_log_weights: dict[tuple[str, str], float],
) -> list[tuple[tuple[str, ...], float]]:
    """All simple directed paths source→endpoint with log-weights (DAG ⇒ simple)."""
    results: list[tuple[tuple[str, ...], float]] = []

    def dfs(node: str, path: list[str], log_w: float) -> None:
        if node == endpoint:
            results.append((tuple(path), log_w))
            return
        for nxt in graph.successors(node):
            dfs(nxt, path + [nxt], log_w + edge_log_weights[(node, nxt)])

    dfs(source, [source], 0.0)
    return results


def _logsumexp(values: list[float]) -> float:
    if not values:
        return NEG_INF
    m = max(values)
    if m == NEG_INF:
        return NEG_INF
    return m + float(np.log(np.sum(np.exp(np.asarray(values) - m))))


def test_diamond_uniform_paths_each_half():
    g = nx.DiGraph([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    weights = {e: 0.0 for e in g.edges}
    ens = FinitePathEnsemble(g, weights)
    ranked = top_k_paths(ens, "A", "D", k=10)
    assert len(ranked) == 2
    by_nodes = {p.nodes: p.probability for p in ranked}
    assert abs(by_nodes[("A", "B", "D")] - 0.5) < 1e-8
    assert abs(by_nodes[("A", "C", "D")] - 0.5) < 1e-8
    assert abs(sum(by_nodes.values()) - 1.0) < 1e-8


def test_dp_matches_enumeration_on_weighted_dag():
    g = nx.DiGraph()
    edges = [
        ("S", "A"),
        ("S", "B"),
        ("A", "C"),
        ("B", "C"),
        ("A", "D"),
        ("C", "E"),
        ("D", "E"),
        ("B", "E"),
    ]
    g.add_edges_from(edges)
    rng = np.random.default_rng(0)
    weights = {e: float(rng.normal()) for e in edges}

    ens = FinitePathEnsemble(g, weights)
    log_z_dp = ens.backward_log_partition("E")["S"]

    enumerated = _enumerate_paths(g, "S", "E", weights)
    log_z_enum = _logsumexp([lw for _, lw in enumerated])
    assert abs(log_z_dp - log_z_enum) < 1e-8

    # Path probabilities from DP top-k (all paths) vs enumeration.
    ranked = top_k_paths(ens, "S", "E", k=100)
    assert abs(sum(p.probability for p in ranked) - 1.0) < 1e-8

    enum_probs = {
        path: float(np.exp(lw - log_z_enum)) for path, lw in enumerated
    }
    for p in ranked:
        assert abs(p.probability - enum_probs[p.nodes]) < 1e-8


def test_path_probabilities_sum_to_one_under_temperature():
    g = nx.DiGraph([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    weights = {
        ("A", "B"): 1.0,
        ("A", "C"): 0.0,
        ("B", "D"): 0.5,
        ("C", "D"): 0.5,
    }
    for temperature in (0.5, 1.0, 2.0):
        ens = FinitePathEnsemble(g, weights, temperature=temperature)
        ranked = top_k_paths(ens, "A", "D", k=10)
        assert abs(sum(p.probability for p in ranked) - 1.0) < 1e-8


def test_edge_marginals_match_enumeration():
    g = nx.DiGraph([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    weights = {
        ("A", "B"): 0.0,
        ("A", "C"): 0.0,
        ("B", "D"): 0.0,
        ("C", "D"): 0.0,
    }
    ens = FinitePathEnsemble(g, weights)
    marg = edge_marginals(ens, "A", "D")
    # Each edge sits on exactly one of two equal paths.
    assert abs(marg[("A", "B")] - 0.5) < 1e-8
    assert abs(marg[("A", "C")] - 0.5) < 1e-8
    assert abs(marg[("B", "D")] - 0.5) < 1e-8
    assert abs(marg[("C", "D")] - 0.5) < 1e-8


def test_node_marginals_and_mass_conservation():
    g = nx.DiGraph([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    ens = FinitePathEnsemble(g, {e: 0.0 for e in g.edges})
    nodes = node_marginals(ens, "A", "D")
    assert abs(nodes["A"] - 1.0) < 1e-8
    assert abs(nodes["D"] - 1.0) < 1e-8
    assert abs(nodes["B"] - 0.5) < 1e-8
    assert abs(nodes["C"] - 0.5) < 1e-8

    # Outflow mass from A equals 1.
    edges = edge_marginals(ens, "A", "D")
    assert abs(edges[("A", "B")] + edges[("A", "C")] - 1.0) < 1e-8


def test_enumeration_order_independence():
    edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("B", "C")]
    # B→C creates an extra path A→B→C→D; still a DAG.
    weights = {e: 0.1 * i for i, e in enumerate(edges)}

    probs_list = []
    for perm in itertools.islice(itertools.permutations(edges), 0, 12):
        g = nx.DiGraph()
        g.add_edges_from(perm)
        ens = FinitePathEnsemble(g, weights)
        ranked = top_k_paths(ens, "A", "D", k=20)
        probs_list.append({p.nodes: p.probability for p in ranked})

    ref = probs_list[0]
    for probs in probs_list[1:]:
        assert set(probs) == set(ref)
        for path in ref:
            assert abs(probs[path] - ref[path]) < 1e-10
