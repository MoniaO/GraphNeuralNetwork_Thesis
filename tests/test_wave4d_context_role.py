"""Wave 4D anti-leakage + structural context selection tests."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest
import torch
from hydra import compose, initialize_config_dir

from experiments.motif_completion import hide_tasks
from hcr.attach_wave4d import pad_context_features, wave4d_enabled
from hcr.context_roles import (
    build_collider_records,
    build_confounder_records,
    build_context_role_registry,
    build_descendant_records,
    build_mediator_records,
    classify_selected_z_role,
)
from hcr.motif_registry_v3 import DUAL_GATES, GATE_OUTCOMES
from hcr.structural_context import (
    StructuralSelectorAudit,
    select_structural_contexts,
)


ROOT = Path(__file__).resolve().parents[1]


def _cfg(hide: str = "parent_a", variant: str = "structural_latent_pairwise"):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(
            config_name="config",
            overrides=[
                "model=TaskA_hgt_hcr",
                f"hcr={variant}",
                "experiment.wave=WAVE4D_CONTEXT_ROLE_AUDIT",
                "experiment.motif_completion.enabled=true",
                f"experiment.motif_completion.hide={hide}",
                "experiment.causal_role_audit.enabled=true",
                "experiment.context_selection.mode=structural",
                "experiment.context_selection.source_graph=train",
                "experiment.context_selection.use_true_graph=false",
                "wandb.enabled=false",
            ],
        )


def test_wave4d_enabled_flag():
    cfg = _cfg()
    assert wave4d_enabled(cfg)


def test_selector_audit_flags():
    audit = StructuralSelectorAudit()
    assert audit.graph_source == "train"
    assert not audit.has_access_to_true_graph
    assert not audit.uses_test_labels


def test_dual_mask_task_counts():
    cfg_a = _cfg("parent_a")
    cfg_b = _cfg("parent_b")
    ta = hide_tasks(cfg_a)
    tb = hide_tasks(cfg_b)
    assert len(ta) == 10
    assert len(tb) == 10
    # Different held-out parents.
    assert {(t.candidate_source, t.candidate_target) for t in ta} != {
        (t.candidate_source, t.candidate_target) for t in tb
    }


def test_structural_selects_coparent_on_toy_graph():
    g = nx.DiGraph()
    # A→G←B, G→Y  (after hide A→G, train still has B→G)
    g.add_edges_from([("B", "G"), ("G", "Y")])
    meta = {
        "A": {"node_type": "drug_exposure", "layer": "2_drugs", "layer_rank": 1},
        "B": {"node_type": "drug_exposure", "layer": "2_drugs", "layer_rank": 1},
        "G": {
            "node_type": "adr_or_intermediate_state",
            "layer": "4_intermediate_states",
            "layer_rank": 3,
        },
        "Y": {
            "node_type": "adr_or_intermediate_state",
            "layer": "6_endpoints",
            "layer_rank": 5,
        },
    }
    ctx = select_structural_contexts(g, "A", "G", "Y", meta)
    assert ctx == ["B"]


def test_structural_rejects_descendant_and_mediator():
    g = nx.DiGraph()
    g.add_edges_from([("A", "Z"), ("Z", "G"), ("B", "G"), ("G", "Y"), ("Y", "Post")])
    meta = {
        n: {
            "node_type": "drug_exposure",
            "layer": "2_drugs",
            "layer_rank": 1,
        }
        for n in ("A", "B", "Z")
    }
    meta["G"] = {
        "node_type": "adr_or_intermediate_state",
        "layer": "4_intermediate_states",
        "layer_rank": 3,
    }
    meta["Y"] = {
        "node_type": "adr_or_intermediate_state",
        "layer": "6_endpoints",
        "layer_rank": 5,
    }
    meta["Post"] = {
        "node_type": "adr_or_intermediate_state",
        "layer": "6_endpoints",
        "layer_rank": 5,
    }
    # Z is mediator A→Z→G — excluded; Post is not a parent of G.
    ctx = select_structural_contexts(g, "A", "G", "Y", meta)
    assert "Z" not in ctx
    assert "Post" not in ctx
    assert "B" in ctx


def test_pad_context_features_17_to_24():
    x = torch.randn(5, 17)
    y = pad_context_features(x, 24)
    assert y.shape == (5, 24)
    assert torch.allclose(y[:, :17], x)
    assert torch.allclose(y[:, 17:], torch.zeros(5, 7))


def test_role_registry_builders_nonempty_on_truth_if_available():
    try:
        cfg = _cfg()
        reg = build_context_role_registry(cfg, max_per_role=20, seed=20260722)
    except FileNotFoundError:
        pytest.skip("dataset_v3 not available")
    roles = {r.role for r in reg}
    assert "true_coparent" in roles
    # At least some hard-negative roles should appear on the audited DAG.
    assert roles & {"confounder", "mediator", "collider", "descendant"}


def test_classify_selected_z_roles():
    g = nx.DiGraph()
    g.add_edges_from(
        [
            ("A", "G"),
            ("B", "G"),
            ("G", "Y"),
            ("Y", "Post"),
            ("U", "A"),
            ("U", "X"),
            ("A", "M"),
            ("M", "X"),
            ("P1", "C"),
            ("P2", "C"),
        ]
    )
    assert classify_selected_z_role(g, "A", "G", "B", "Y") == "true_coparent"
    assert classify_selected_z_role(g, "A", "X", "U", None) == "confounder"
    assert classify_selected_z_role(g, "A", "X", "M", None) == "mediator"
    assert classify_selected_z_role(g, "P1", "P2", "C", None) == "collider"
    assert classify_selected_z_role(g, "A", "G", "Post", "Y") == "descendant"


def test_gate_outcomes_cover_duals():
    for gate in DUAL_GATES:
        assert gate in GATE_OUTCOMES


def test_role_constructors_smoke():
    g = nx.DiGraph()
    g.add_edges_from(
        [
            ("A", "G"),
            ("B", "G"),
            ("G", "Y"),
            ("Y", "Post"),
            ("U", "X"),
            ("U", "Z"),
            ("X", "M"),
            ("M", "Z"),
            ("P1", "C"),
            ("P2", "C"),
        ]
    )
    assert build_confounder_records(g)
    assert build_mediator_records(g)
    assert build_collider_records(g)
    assert build_descendant_records(g)
