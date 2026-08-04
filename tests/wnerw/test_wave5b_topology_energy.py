"""Unit tests for Wave 5B topology/energy helpers."""

from __future__ import annotations

import math

import networkx as nx
import pandas as pd

from wnerw.wave5b.energy_maps import (
    EnergyConfig,
    add_matched_edgewise_shuffle,
    build_edge_energies,
    uniform_edge_scores,
)
from wnerw.wave5b.exact_metrics import (
    backward_log_partition,
    exact_path_entropy,
    true_path_mass,
)
from wnerw.wave5b.graph_builders import selected_edge_mask


def test_selected_edge_mask_keeps_train_even_if_topo_false():
    edges = pd.DataFrame(
        {
            "source": ["age", "A", "B"],
            "target": ["frailty", "G", "G"],
            "edge_in_train": [True, False, False],
            "topological_allowed": [False, True, True],
            "p": [0.1, 0.9, 0.1],
        }
    )
    mask = selected_edge_mask(edges, "p", 0.5)
    assert bool(mask.iloc[0]) is True
    assert bool(mask.iloc[1]) is True
    assert bool(mask.iloc[2]) is False


def test_uniform_scores_give_equal_two_path_mass():
    g = nx.DiGraph([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    scores = uniform_edge_scores(g)
    back = backward_log_partition(g, scores, "D")
    m1 = true_path_mass(["A", "B", "D"], g, scores, back)
    m2 = true_path_mass(["A", "C", "D"], g, scores, back)
    assert abs(m1 - 0.5) < 1e-12
    assert abs(m2 - 0.5) < 1e-12
    ent = exact_path_entropy(g, scores, back, "A", "D")
    assert abs(ent - math.log(2)) < 1e-9


def test_energy_train_edge_log_evidence_zero():
    g = nx.DiGraph([("A", "B"), ("B", "C")])
    edges = pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["B", "C"],
            "edge_in_train": [True, False],
            "p_hcr2_calibrated": [0.99, 0.8],
            "hcr_uncertainty": [0.5, 0.25],
        }
    )
    cfg = EnergyConfig(
        probability_column="p_hcr2_calibrated",
        temperature=1.0,
        length_penalty=0.1,
        uncertainty_penalty=0.5,
    )
    scores = build_edge_energies(g, edges, cfg)
    # train: 0 - 0 - 0.1
    assert abs(scores[("A", "B")] - (-0.1)) < 1e-12
    # predicted: log(0.8) - 0.5*0.25 - 0.1
    expected = math.log(0.8) - 0.125 - 0.1
    assert abs(scores[("B", "C")] - expected) < 1e-12


def test_matched_shuffle_preserves_multiset():
    edges = pd.DataFrame(
        {
            "source": [f"s{i}" for i in range(6)],
            "target": [f"t{i}" for i in range(6)],
            "edge_in_train": [False] * 6,
            "topological_allowed": [True] * 6,
            "edge_type": ["drug_effect"] * 6,
            "p_hcr2_calibrated": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "hcr_uncertainty": [0.1, 0.1, 0.5, 0.5, 0.9, 0.9],
        }
    )
    out = add_matched_edgewise_shuffle(
        edges, "p_hcr2_calibrated", "p_shuf", seed=0
    )
    assert sorted(out["p_shuf"].tolist()) == sorted(
        edges["p_hcr2_calibrated"].tolist()
    )
