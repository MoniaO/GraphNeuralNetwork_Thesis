"""Shared Wave 5C runtime helpers (graph + patients + variants)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from hydra import compose, initialize_config_dir

from hcr.motifs import load_truth_graph
from hcr.structural_context import load_node_metadata, matched_random_context
from wnerw.wave5b.graph_builders import (
    build_model_graph,
    hub_nodes_from_degree,
)
from wnerw.wave5b.io import load_edges, resolve_path
from wnerw.wave5c.edge_context_registry import (
    admissible_patient_feature_table,
    build_context_map,
    build_edge_context_registry,
    g_train_from_evidence_and_hetero,
)
from wnerw.wave5c.patient_activity import build_patient_activity
from wnerw.wave5c.patient_energy import PatientEnergyConfig


ROOT = Path(__file__).resolve().parents[3]


def load_cfg(path: Path | None = None) -> dict:
    path = path or (ROOT / "configs/wnerw/wave5c.yaml")
    return yaml.safe_load(path.read_text())


def compose_task_cfg(seed: int):
    overrides = [
        "model=TaskA_hgt",
        "hcr=none",
        "model.num_layers=1",
        "model.heads=8",
        f"training.seed={seed}",
        "training.device=cpu",
        "data.dataset.scenario=clean",
        "data.feature_ablation_profile=empirical",
        "data.candidate_seed=20260722",
        "wandb.enabled=false",
        "experiment.wave=WAVE5C",
        "experiment.motif_completion.enabled=false",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name="config", overrides=overrides)


def load_seed_bundle(cfg: dict, seed: int) -> dict[str, Any]:
    from data.PreprocessingTaskA.load_hetero_recon_data import load_recon_heterodata
    from data.patient_matrix import load_patient_matrix_with_split

    edge_path = resolve_path(cfg["paths"]["edge_pattern"], seed, ROOT)
    edges = load_edges(edge_path)
    task_cfg = compose_task_cfg(seed)
    train, _, _, _ = load_recon_heterodata(task_cfg)
    g_train = g_train_from_evidence_and_hetero(train)
    meta = load_node_metadata(task_cfg)
    nodes, _ = load_truth_graph(task_cfg)

    thr = float(cfg["graph_selection"]["hcr_threshold"])
    graph_hcr = build_model_graph(
        edges,
        cfg["columns"]["hcr_probability"],
        thr,
        nodes=nodes,
    )
    graph_hgt = build_model_graph(
        edges,
        cfg["columns"]["hgt_probability"],
        float(cfg["graph_selection"]["hgt_threshold"]),
        nodes=nodes,
    )

    registry = build_edge_context_registry(
        g_train,
        edges,
        meta,
        seed=seed,
        working_graph=graph_hcr,
    )
    context_map = build_context_map(registry)
    admissible_df = admissible_patient_feature_table(meta)
    admissible = set(admissible_df.loc[admissible_df["admissible"], "node"].astype(str))

    patients = load_patient_matrix_with_split(task_cfg)
    hubs = hub_nodes_from_degree(
        graph_hcr, degree_quantile=float(cfg["hub_definition"]["degree_quantile"])
    )

    return {
        "seed": seed,
        "edges": edges,
        "graph_hcr": graph_hcr,
        "graph_hgt": graph_hgt,
        "g_train": g_train,
        "meta": meta,
        "nodes": nodes,
        "registry": registry,
        "context_map": context_map,
        "admissible": admissible,
        "admissible_df": admissible_df,
        "patients": patients,
        "hubs": hubs,
        "task_cfg": task_cfg,
    }


def energy_variants(
    alpha: float,
    eta: float,
    *,
    beta_c3: float = 0.1,
) -> dict[str, PatientEnergyConfig]:
    base = dict(temperature=0.5, length_penalty=0.1)
    return {
        "C0_neutral": PatientEnergyConfig(
            **base, node_activity_weight=0.0, gate_weight=0.0, static_hcr_weight=0.0
        ),
        "C1_node_activity": PatientEnergyConfig(
            **base, node_activity_weight=alpha, gate_weight=0.0, static_hcr_weight=0.0
        ),
        "C2_structural_gate": PatientEnergyConfig(
            **base, node_activity_weight=alpha, gate_weight=eta, static_hcr_weight=0.0
        ),
        "C3_gate_plus_hcr_prior": PatientEnergyConfig(
            **base,
            node_activity_weight=alpha,
            gate_weight=eta,
            static_hcr_weight=beta_c3,
        ),
        "C4_patient_shuffle": PatientEnergyConfig(
            **base, node_activity_weight=alpha, gate_weight=eta, static_hcr_weight=0.0
        ),
        "C5_context_shuffle": PatientEnergyConfig(
            **base, node_activity_weight=alpha, gate_weight=eta, static_hcr_weight=0.0
        ),
        "C6_matched_random_context": PatientEnergyConfig(
            **base, node_activity_weight=alpha, gate_weight=eta, static_hcr_weight=0.0
        ),
        "C7_hgt_topology": PatientEnergyConfig(
            **base, node_activity_weight=alpha, gate_weight=eta, static_hcr_weight=0.0
        ),
    }


def patient_activity_for(
    patients: pd.DataFrame,
    patient_id: str,
    admissible: set[str],
) -> dict[str, float]:
    row = patients.loc[patients["patient_id"].astype(str) == str(patient_id)]
    if row.empty:
        raise KeyError(patient_id)
    return build_patient_activity(row.iloc[0], admissible)


def shuffle_patient_profiles(patient_ids: list[str], seed: int) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(np.asarray(patient_ids, dtype=object))
    return dict(zip(patient_ids, [str(x) for x in shuffled]))


def matched_random_context_map(
    context_map: dict[tuple[str, str], tuple[str, ...]],
    meta: dict,
    seed: int,
) -> dict[tuple[str, str], tuple[str, ...]]:
    """Replace each structural context with a matched-random substitute."""
    out: dict[tuple[str, str], tuple[str, ...]] = {}
    for (src, tgt), ctxs in context_map.items():
        new_ctx = []
        for i, c in enumerate(ctxs):
            z = matched_random_context(
                src,
                tgt,
                None,
                c,
                meta,
                int(seed) + i,
            )
            new_ctx.append(str(z) if z is not None else str(c))
        out[(src, tgt)] = tuple(new_ctx)
    return out
