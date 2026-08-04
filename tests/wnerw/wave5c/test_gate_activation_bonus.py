import networkx as nx
import numpy as np
import pandas as pd

from wnerw.wave5b.exact_metrics import backward_log_partition
from wnerw.wave5c.edge_marginals import exact_edge_mass
from wnerw.wave5c.patient_activity import gate_activation
from wnerw.wave5c.patient_energy import PatientEnergyConfig, build_patient_edge_scores


def test_gate_activation_and_table():
    assert gate_activation("A", ("B",), {"A": 0, "B": 0}) == 0.0
    assert gate_activation("A", ("B",), {"A": 1, "B": 0}) == 0.0
    assert gate_activation("A", ("B",), {"A": 0, "B": 1}) == 0.0
    assert gate_activation("A", ("B",), {"A": 1, "B": 1}) == 1.0


def test_active_context_increases_hidden_edge_mass():
    g = nx.DiGraph([("A", "G"), ("B", "G"), ("G", "Y"), ("A", "Y")])
    edges = pd.DataFrame(
        {
            "source": ["A", "B", "G", "A"],
            "target": ["G", "G", "Y", "Y"],
            "edge_in_train": [False, True, True, True],
            "p_hcr2_calibrated": [0.5, 0.5, 0.5, 0.5],
        }
    )
    cfg = PatientEnergyConfig(
        temperature=0.5,
        length_penalty=0.1,
        node_activity_weight=0.0,
        gate_weight=1.0,
        static_hcr_weight=0.0,
    )
    cmap = {("A", "G"): ("B",)}
    off = build_patient_edge_scores(
        g, edges, {"A": 1.0, "B": 0.0}, set(), cmap, cfg
    )
    on = build_patient_edge_scores(
        g, edges, {"A": 1.0, "B": 1.0}, set(), cmap, cfg
    )
    back_off = backward_log_partition(g, off, "Y")
    back_on = backward_log_partition(g, on, "Y")
    m_off = exact_edge_mass(("A", "G"), g, off, back_off, "A")
    m_on = exact_edge_mass(("A", "G"), g, on, back_on, "A")
    assert m_on > m_off
