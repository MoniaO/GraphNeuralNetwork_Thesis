"""Backward partition sanity checks for FinitePathEnsemble."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from wnerw.finite_path_ensemble import (
    FinitePathEnsemble,
    logsumexp,
    transition_probabilities,
)
from wnerw.types import NEG_INF


def _diamond(weights: dict[tuple[str, str], float] | None = None) -> FinitePathEnsemble:
    g = nx.DiGraph()
    g.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    if weights is None:
        weights = {e: 0.0 for e in g.edges}
    return FinitePathEnsemble(g, weights)


def test_logsumexp_empty_and_neg_inf():
    assert logsumexp([]) == NEG_INF
    assert logsumexp([NEG_INF, NEG_INF]) == NEG_INF
    assert abs(logsumexp([0.0, 0.0]) - math.log(2.0)) < 1e-12


def test_rejects_cycles():
    g = nx.DiGraph([("A", "B"), ("B", "A")])
    with pytest.raises(ValueError, match="DAG"):
        FinitePathEnsemble(g, {("A", "B"): 0.0, ("B", "A"): 0.0})


def test_rejects_nonpositive_temperature():
    g = nx.DiGraph([("A", "B")])
    with pytest.raises(ValueError, match="temperature"):
        FinitePathEnsemble(g, {("A", "B"): 0.0}, temperature=0.0)


def test_endpoint_has_zero_log_partition():
    ens = _diamond()
    log_z = ens.backward_log_partition("D")
    assert log_z["D"] == 0.0


def test_unreachable_node_has_neg_inf():
    g = nx.DiGraph([("A", "B"), ("C", "D")])
    ens = FinitePathEnsemble(
        g, {("A", "B"): 0.0, ("C", "D"): 0.0}
    )
    log_z = ens.backward_log_partition("D")
    assert log_z["D"] == 0.0
    assert log_z["C"] == 0.0  # one-step path C→D with weight 0
    assert log_z["A"] == NEG_INF
    assert log_z["B"] == NEG_INF


def test_diamond_uniform_log_partition():
    # Two paths of weight 1 each → logZ(A) = log(2)
    ens = _diamond()
    log_z = ens.backward_log_partition("D")
    assert abs(log_z["A"] - math.log(2.0)) < 1e-12
    assert abs(log_z["B"] - 0.0) < 1e-12
    assert abs(log_z["C"] - 0.0) < 1e-12


def test_transitions_sum_to_one():
    ens = _diamond()
    log_z = ens.backward_log_partition("D")
    probs = transition_probabilities(ens, "A", "D", log_z)
    assert set(probs) == {"B", "C"}
    assert abs(sum(probs.values()) - 1.0) < 1e-12
    assert abs(probs["B"] - 0.5) < 1e-12
    assert abs(probs["C"] - 0.5) < 1e-12


def test_weighted_transitions_prefer_heavier_branch():
    weights = {
        ("A", "B"): math.log(3.0),
        ("A", "C"): math.log(1.0),
        ("B", "D"): 0.0,
        ("C", "D"): 0.0,
    }
    ens = _diamond(weights)
    probs = transition_probabilities(ens, "A", "D")
    assert abs(probs["B"] - 0.75) < 1e-12
    assert abs(probs["C"] - 0.25) < 1e-12


def test_forward_matches_backward_partition():
    ens = _diamond({("A", "B"): 0.2, ("A", "C"): -0.1, ("B", "D"): 0.5, ("C", "D"): 0.3})
    assert abs(
        ens.forward_log_partition("A")["D"]
        - ens.backward_log_partition("D")["A"]
    ) < 1e-10


def test_result_independent_of_edge_insertion_order():
    edges_a = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]
    edges_b = list(reversed(edges_a))
    w = {e: float(i) * 0.1 for i, e in enumerate(edges_a)}

    g1 = nx.DiGraph()
    g1.add_edges_from(edges_a)
    g2 = nx.DiGraph()
    g2.add_edges_from(edges_b)

    z1 = FinitePathEnsemble(g1, w).backward_log_partition("D")
    z2 = FinitePathEnsemble(g2, w).backward_log_partition("D")
    for node in ("A", "B", "C", "D"):
        assert abs(z1[node] - z2[node]) < 1e-12
