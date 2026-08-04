"""Dedicated top-K path tests."""

from __future__ import annotations

import math

import networkx as nx

from wnerw.finite_path_ensemble import FinitePathEnsemble
from wnerw.topk_paths import top_k_paths


def test_topk_orders_by_weight():
    g = nx.DiGraph([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    weights = {
        ("A", "B"): math.log(3.0),
        ("A", "C"): math.log(1.0),
        ("B", "D"): 0.0,
        ("C", "D"): 0.0,
    }
    ens = FinitePathEnsemble(g, weights)
    ranked = top_k_paths(ens, "A", "D", k=2)
    assert ranked[0].nodes == ("A", "B", "D")
    assert abs(ranked[0].probability - 0.75) < 1e-12
    assert ranked[1].nodes == ("A", "C", "D")


def test_topk_respects_k():
    g = nx.DiGraph([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    ens = FinitePathEnsemble(g, {e: 0.0 for e in g.edges})
    assert len(top_k_paths(ens, "A", "D", k=1)) == 1
