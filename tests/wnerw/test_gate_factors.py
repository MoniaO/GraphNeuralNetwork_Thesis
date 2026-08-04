"""Gate potential unit tests."""

from __future__ import annotations

import math

from wnerw.gate_factors import (
    GateSpec,
    apply_gate_potentials_to_edge_weights,
    gate_potential,
    population_gate_scores_from_d1,
)
from wnerw.potentials import build_edge_log_weights, safe_logit


def test_gate_added_once_on_incoming_edges():
    gate = GateSpec(
        gate_node="G",
        required_inputs=("A", "B"),
        outcome_nodes=("Y",),
    )
    score = gate_potential(gate, motif_score=1.5, mode="population")
    assert score == 1.5
    weights = {("A", "G"): 0.0, ("B", "G"): 0.0, ("G", "Y"): 0.0}
    updated = apply_gate_potentials_to_edge_weights(weights, {"G": score})
    assert updated[("A", "G")] == 1.5
    assert updated[("B", "G")] == 1.5
    assert updated[("G", "Y")] == 0.0  # not re-added on leaving the gate


def test_hard_gate_inactive_is_neg_inf():
    gate = GateSpec("G", ("A", "B"))
    assert gate_potential(gate, active_inputs={"A"}, mode="hard") == float("-inf")
    assert gate_potential(gate, active_inputs={"A", "B"}, motif_score=2.0, mode="hard") == 2.0


def test_soft_gate_penalty():
    gate = GateSpec("G", ("A", "B"))
    assert gate_potential(
        gate, active_inputs=set(), mode="soft", inactive_penalty=5.0
    ) == -5.0


def test_population_gate_scores_mean_log_d1():
    rows = [
        {"source": "A", "target": "G", "p_d1_calibrated": 0.8},
        {"source": "B", "target": "G", "p_d1_calibrated": 0.2},
    ]
    gates = {"G": GateSpec("G", ("A", "B"))}
    scores = population_gate_scores_from_d1(rows, gates=gates, coefficient=1.0)
    expected = 0.5 * (math.log(0.8) + math.log(0.2))
    assert abs(scores["G"] - expected) < 1e-12


def test_d1_log_prob_and_delta_logit_modes():
    edges = [
        {
            "source": "A",
            "target": "G",
            "edge_in_train": False,
            "p_hgt_calibrated": 0.4,
            "p_d1_calibrated": 0.9,
            "hcr_uncertainty": 0.0,
        }
    ]
    w_log = build_edge_log_weights(edges, mode="d1_log_prob", length_penalty=0.0)
    assert abs(w_log[("A", "G")] - math.log(0.9)) < 1e-12

    w_delta = build_edge_log_weights(
        edges, mode="d1_delta_logit", gamma_delta=1.0, length_penalty=0.0
    )
    expected = math.log(0.4) + (safe_logit(0.9) - safe_logit(0.4))
    assert abs(w_delta[("A", "G")] - expected) < 1e-12

    w_nested = build_edge_log_weights(edges, mode="d1_nested", length_penalty=0.0)
    assert abs(w_nested[("A", "G")] - math.log(0.9)) < 1e-12
