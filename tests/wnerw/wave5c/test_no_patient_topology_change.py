import networkx as nx

from wnerw.wave5c.patient_energy import PatientEnergyConfig, build_patient_edge_scores


def test_topology_independent_of_patient():
    g = nx.DiGraph([("A", "G"), ("B", "G"), ("G", "Y")])
    edges = __import__("pandas").DataFrame(
        {
            "source": ["A", "B", "G"],
            "target": ["G", "G", "Y"],
            "edge_in_train": [False, True, True],
            "p_hcr2_calibrated": [0.9, 0.9, 0.9],
        }
    )
    cfg = PatientEnergyConfig(gate_weight=1.0, node_activity_weight=0.25)
    s1 = build_patient_edge_scores(
        g, edges, {"A": 1, "B": 0}, {"A", "B"}, {("A", "G"): ("B",)}, cfg
    )
    s2 = build_patient_edge_scores(
        g, edges, {"A": 1, "B": 1}, {"A", "B"}, {("A", "G"): ("B",)}, cfg
    )
    assert set(s1) == set(s2) == set(g.edges)
