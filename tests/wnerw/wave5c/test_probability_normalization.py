import math

import networkx as nx

from wnerw.wave5b.exact_metrics import backward_log_partition, path_log_score
from wnerw.wave5c.patient_energy import PatientEnergyConfig, build_patient_edge_scores
import pandas as pd


def test_path_probabilities_sum_to_one():
    g = nx.DiGraph([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    edges = pd.DataFrame(
        {
            "source": ["A", "A", "B", "C"],
            "target": ["B", "C", "D", "D"],
            "edge_in_train": [True] * 4,
            "p_hcr2_calibrated": [0.5] * 4,
        }
    )
    cfg = PatientEnergyConfig()
    scores = build_patient_edge_scores(g, edges, {}, set(), {}, cfg)
    back = backward_log_partition(g, scores, "D")
    log_z = back["A"]
    mass = 0.0
    for path in nx.all_simple_paths(g, "A", "D"):
        mass += math.exp(path_log_score(path, scores) - log_z)
    assert abs(mass - 1.0) < 1e-9
