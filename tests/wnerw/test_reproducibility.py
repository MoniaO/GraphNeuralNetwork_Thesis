"""Determinism / reproducibility of the finite-path engine."""

from __future__ import annotations

import networkx as nx

from wnerw.finite_path_ensemble import FinitePathEnsemble
from wnerw.path_marginals import edge_marginals
from wnerw.topk_paths import top_k_paths


def test_repeated_calls_identical():
    g = nx.DiGraph([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    w = {("A", "B"): 0.3, ("A", "C"): -0.2, ("B", "D"): 0.1, ("C", "D"): 0.4}
    ens = FinitePathEnsemble(g, w)
    a = top_k_paths(ens, "A", "D", k=5)
    b = top_k_paths(ens, "A", "D", k=5)
    assert [p.nodes for p in a] == [p.nodes for p in b]
    assert [p.probability for p in a] == [p.probability for p in b]
    assert edge_marginals(ens, "A", "D") == edge_marginals(ens, "A", "D")
