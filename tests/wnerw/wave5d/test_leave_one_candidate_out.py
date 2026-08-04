"""Candidate edge must not provide its own path-completion evidence."""

from __future__ import annotations

import networkx as nx
import numpy as np

from link_prediction.wave5d.forward_backward_support import (
    candidate_path_completion_score,
    graph_without_candidate,
)


def test_candidate_removed_from_support_graph():
    g = nx.DiGraph()
    g.add_edges_from([("s", "u"), ("u", "v"), ("v", "y"), ("s", "y")])
    g2 = graph_without_candidate(g, ("u", "v"))
    assert g.has_edge("u", "v")
    assert not g2.has_edge("u", "v")


def test_completion_does_not_use_candidate_edge_weight():
    g = nx.DiGraph()
    g.add_edges_from([("s", "u"), ("u", "v"), ("v", "y")])
    scores = {e: 1.0 for e in g.edges}
    # Inflate candidate weight — should not matter after leave-one-out.
    scores[("u", "v")] = 100.0
    c1 = candidate_path_completion_score(
        g, ["s"], ["y"], ("u", "v"), scores
    )
    scores[("u", "v")] = -100.0
    c2 = candidate_path_completion_score(
        g, ["s"], ["y"], ("u", "v"), scores
    )
    assert np.isfinite(c1) and np.isfinite(c2)
    assert np.isclose(c1, c2)
