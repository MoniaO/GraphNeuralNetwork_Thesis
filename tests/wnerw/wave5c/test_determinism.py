import networkx as nx
import pandas as pd

from wnerw.wave5c.patient_energy import PatientEnergyConfig, build_patient_edge_scores


def test_scores_deterministic():
    g = nx.DiGraph([("A", "G"), ("G", "Y")])
    edges = pd.DataFrame(
        {
            "source": ["A", "G"],
            "target": ["G", "Y"],
            "edge_in_train": [False, True],
            "p_hcr2_calibrated": [0.8, 0.8],
        }
    )
    cfg = PatientEnergyConfig(gate_weight=1.0)
    a = {"A": 1.0, "B": 1.0}
    s1 = build_patient_edge_scores(g, edges, a, set(), {("A", "G"): ("B",)}, cfg)
    s2 = build_patient_edge_scores(g, edges, a, set(), {("A", "G"): ("B",)}, cfg)
    assert s1 == s2


def test_neutral_identical_across_patients():
    g = nx.DiGraph([("A", "G"), ("G", "Y")])
    edges = pd.DataFrame(
        {
            "source": ["A", "G"],
            "target": ["G", "Y"],
            "edge_in_train": [False, True],
            "p_hcr2_calibrated": [0.8, 0.8],
        }
    )
    cfg = PatientEnergyConfig(
        node_activity_weight=0.0, gate_weight=0.0, static_hcr_weight=0.0
    )
    s1 = build_patient_edge_scores(
        g, edges, {"A": 1, "B": 0}, {"A", "B"}, {("A", "G"): ("B",)}, cfg
    )
    s2 = build_patient_edge_scores(
        g, edges, {"A": 1, "B": 1}, {"A", "B"}, {("A", "G"): ("B",)}, cfg
    )
    assert s1 == s2
