"""Edge / node marginal tests."""

from __future__ import annotations

import networkx as nx

from wnerw.finite_path_ensemble import FinitePathEnsemble
from wnerw.path_marginals import edge_marginals, node_marginals


def test_edge_marginals_flow_conservation_diamond():
    g = nx.DiGraph([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    ens = FinitePathEnsemble(g, {e: 0.0 for e in g.edges})
    m = edge_marginals(ens, "A", "D")
    assert abs(m[("A", "B")] + m[("A", "C")] - 1.0) < 1e-8
    assert abs(m[("B", "D")] + m[("C", "D")] - 1.0) < 1e-8
    assert abs(m[("A", "B")] - m[("B", "D")]) < 1e-8


def test_node_marginals_bounded():
    g = nx.DiGraph([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    ens = FinitePathEnsemble(g, {e: 0.0 for e in g.edges})
    nodes = node_marginals(ens, "A", "D")
    for mass in nodes.values():
        assert 0.0 <= mass <= 1.0 + 1e-12
