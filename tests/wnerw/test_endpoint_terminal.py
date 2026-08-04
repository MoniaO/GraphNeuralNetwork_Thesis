"""Endpoint must be terminal for the finite-path ensemble."""

from __future__ import annotations

import math

import networkx as nx
import pytest

from wnerw.finite_path_ensemble import FinitePathEnsemble, transition_probabilities
from wnerw.topk_paths import top_k_paths
from wnerw.types import NEG_INF


def test_endpoint_partition_fixed_at_zero_even_with_outgoing():
    """Outgoing edges from the endpoint are ignored: e is absorbing/terminal."""
    g = nx.DiGraph([("A", "E"), ("E", "X")])
    weights = {("A", "E"): 0.0, ("E", "X"): 5.0}
    ens = FinitePathEnsemble(g, weights)

    log_z = ens.backward_log_partition("E")
    assert log_z["E"] == 0.0
    # Path mass stops at E; E→X must not contribute to Z(A|E).
    assert abs(log_z["A"] - 0.0) < 1e-12


def test_no_transition_out_of_endpoint():
    g = nx.DiGraph([("A", "E"), ("E", "X")])
    ens = FinitePathEnsemble(
        g, {("A", "E"): 0.0, ("E", "X"): 1.0}
    )
    assert transition_probabilities(ens, "E", "E") == {}


def test_paths_end_exactly_at_endpoint():
    g = nx.DiGraph([("A", "B"), ("B", "E"), ("E", "X"), ("B", "X")])
    ens = FinitePathEnsemble(
        g,
        {
            ("A", "B"): 0.0,
            ("B", "E"): 0.0,
            ("E", "X"): 10.0,
            ("B", "X"): 10.0,
        },
    )
    ranked = top_k_paths(ens, "A", "E", k=10)
    assert len(ranked) == 1
    assert ranked[0].nodes == ("A", "B", "E")
    assert abs(ranked[0].probability - 1.0) < 1e-12


def test_unknown_endpoint_raises():
    g = nx.DiGraph([("A", "B")])
    ens = FinitePathEnsemble(g, {("A", "B"): 0.0})
    with pytest.raises(KeyError, match="Unknown endpoint"):
        ens.backward_log_partition("Z")


def test_source_without_route_has_empty_topk():
    g = nx.DiGraph([("A", "B"), ("C", "D")])
    ens = FinitePathEnsemble(
        g, {("A", "B"): 0.0, ("C", "D"): 0.0}
    )
    assert ens.backward_log_partition("D")["A"] == NEG_INF
    assert top_k_paths(ens, "A", "D", k=5) == []
