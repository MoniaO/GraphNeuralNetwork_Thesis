"""Scorer / graph builder must not use G_true edges as potentials."""

from __future__ import annotations

import math

import pandas as pd

from wnerw.graph_builder import build_wnerw_graph
from wnerw.potentials import build_edge_log_weights, nested_edge_log_weight


def test_nested_weight_equals_log_hcr2_when_betas_default():
    q = nested_edge_log_weight(0.2, 0.8, None, uncertainty=0.0)
    assert abs(q - math.log(0.8)) < 1e-12


def test_graph_builder_ignores_true_only_edges_not_in_train_or_predictions():
    nodes = pd.DataFrame(
        {
            "node": ["A", "B", "C", "D"],
            "layer": [
                "2_drugs",
                "3_mechanisms",
                "4_intermediate_states",
                "6_endpoints",
            ],
        }
    )
    train = pd.DataFrame({"source": ["A"], "target": ["B"]})
    # Predicted weak edge below threshold — must not appear.
    predicted = pd.DataFrame(
        {
            "source": ["B", "A"],
            "target": ["D", "C"],
            "p_calibrated": [0.1, 0.9],
        }
    )
    # A "true" shortcut A→D is NOT provided to the builder.
    g = build_wnerw_graph(train, predicted, nodes, threshold=0.5)
    assert g.has_edge("A", "B")
    assert g.has_edge("A", "C")
    assert not g.has_edge("B", "D")
    assert not g.has_edge("A", "D")


def test_train_edges_kept_even_if_same_layer():
    """topological_allowed must not drop G_train edges (Wave 5A.1 contract)."""
    nodes = pd.DataFrame(
        {
            "node": ["age", "frailty", "drug", "endpoint"],
            "layer": [
                "1_patient_context",
                "1_patient_context",
                "2_drugs",
                "6_endpoints",
            ],
        }
    )
    train = pd.DataFrame(
        {
            "source": ["age", "drug"],
            "target": ["frailty", "endpoint"],
        }
    )
    predicted = pd.DataFrame(
        {"source": [], "target": [], "p_calibrated": []}
    )
    g = build_wnerw_graph(train, predicted, nodes, threshold=0.5)
    assert g.has_edge("age", "frailty")
    assert g.has_edge("drug", "endpoint")


def test_build_edge_log_weights_train_edges_near_certain():
    edges = [
        {"source": "A", "target": "B", "edge_in_train": True},
        {
            "source": "B",
            "target": "C",
            "edge_in_train": False,
            "p_hgt": 0.5,
            "p_hcr2": 0.8,
            "hcr_uncertainty": 0.0,
        },
    ]
    w = build_edge_log_weights(edges, length_penalty=0.0)
    assert abs(w[("A", "B")] - math.log(1.0 - 1e-6)) < 1e-9
    assert abs(w[("B", "C")] - math.log(0.8)) < 1e-9
