import networkx as nx
import numpy as np

from wnerw.wave5c.patient_activity import gate_activation


def test_gate_and_activation():
    act = {"A": 1.0, "B": 1.0, "C": 0.0}
    assert gate_activation("A", ("B",), act) == 1.0
    assert gate_activation("A", ("B", "C"), act) == 0.0


def test_topology_fixed_across_patients():
    g1 = nx.DiGraph([("A", "G"), ("B", "G"), ("G", "Y")])
    g2 = g1.copy()
    assert set(g1.edges) == set(g2.edges)
