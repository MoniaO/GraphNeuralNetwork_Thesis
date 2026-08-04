import numpy as np

from wnerw.wave5c.patient_energy import PatientEnergyConfig, build_patient_edge_scores
import networkx as nx
import pandas as pd


def test_gate_bonus_is_added_once():
    score_without_gate = -0.1 / 0.5
    score_with_gate = (-0.1 + 1.0) / 0.5
    assert np.isclose(score_with_gate - score_without_gate, 2.0)

    g = nx.DiGraph([("A", "G"), ("G", "Y")])
    edges = pd.DataFrame(
        {
            "source": ["A", "G"],
            "target": ["G", "Y"],
            "edge_in_train": [False, True],
            "p_hcr2_calibrated": [0.9, 0.9],
        }
    )
    cfg = PatientEnergyConfig(gate_weight=1.0, node_activity_weight=0.0)
    scores = build_patient_edge_scores(
        g,
        edges,
        {"A": 1.0, "B": 1.0},
        set(),
        {("A", "G"): ("B",)},
        cfg,
    )
    # Bonus only on A→G, not on G→Y
    assert np.isclose(scores[("A", "G")], score_with_gate)
    assert np.isclose(scores[("G", "Y")], score_without_gate)
