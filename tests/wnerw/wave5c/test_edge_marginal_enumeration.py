import networkx as nx
import pandas as pd

from wnerw.wave5b.exact_metrics import backward_log_partition
from wnerw.wave5c.edge_marginals import enumerate_edge_mass, exact_edge_mass
from wnerw.wave5c.patient_energy import PatientEnergyConfig, build_patient_edge_scores


def test_dp_matches_enumeration():
    g = nx.DiGraph([("A", "G"), ("A", "C"), ("G", "Y"), ("C", "Y"), ("B", "G")])
    edges = pd.DataFrame(
        {
            "source": ["A", "A", "G", "C", "B"],
            "target": ["G", "C", "Y", "Y", "G"],
            "edge_in_train": [False, True, True, True, True],
            "p_hcr2_calibrated": [0.7] * 5,
        }
    )
    cfg = PatientEnergyConfig(gate_weight=1.0, node_activity_weight=0.0)
    scores = build_patient_edge_scores(
        g, edges, {"A": 1, "B": 1}, set(), {("A", "G"): ("B",)}, cfg
    )
    back = backward_log_partition(g, scores, "Y")
    dp = exact_edge_mass(("A", "G"), g, scores, back, "A")
    en = enumerate_edge_mass(("A", "G"), g, scores, "A", "Y")
    assert abs(dp - en) < 1e-8
