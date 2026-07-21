#!/usr/bin/env python3
"""Validate v2.2 DAG, deterministic interactions, scenarios, and endpoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "2. Data" / "code"))
from build_v2_2_interaction_spec import ADR_BURDEN_COMPONENTS, DRUGS


DATA = ROOT / "2. Data" / "dataset_v2_2"
SCENARIOS = [
    "clean",
    "hidden_confounder",
    "selection_bias",
    "no_overlap",
    "noisy_documentation",
    "multihospital",
]
ENDPOINTS = [
    "AKI",
    "DILI",
    "Depression",
    "Falls",
    "Delirium",
    "GI_bleeding",
    "Hyponatremia",
    "Hyperkalemia",
    "QT_arrhythmia",
    "Hospitalization",
]


def main() -> None:
    nodes = pd.read_csv(DATA / "synthetic_pharmacotherapy_v2_2_nodes.csv")
    edges = pd.read_csv(DATA / "synthetic_pharmacotherapy_v2_2_edges_audited.csv")
    samples = {
        scenario: pd.read_csv(
            DATA / f"synthetic_pharmacotherapy_v2_2_samples_{scenario}.csv"
        )
        for scenario in SCENARIOS
    }
    checks: dict[str, object] = {}
    graph = nx.from_pandas_edgelist(
        edges, source="source", target="target", create_using=nx.DiGraph
    )
    graph.add_nodes_from(nodes["node"])
    checks["node_count"] = len(nodes)
    checks["edge_count"] = len(edges)
    checks["is_dag"] = nx.is_directed_acyclic_graph(graph)
    checks["duplicate_nodes"] = int(nodes["node"].duplicated().sum())
    checks["duplicate_edges"] = int(edges.duplicated(["source", "target"]).sum())
    checks["missing_edge_nodes"] = int(
        (~edges["source"].isin(nodes["node"]) | ~edges["target"].isin(nodes["node"])).sum()
    )
    checks["hospitalization_parent_count"] = int(
        edges["target"].eq("Hospitalization").sum()
    )

    clean = samples["clean"]
    expected = {
        "active_drug_count": clean[DRUGS].sum(axis=1).to_numpy(),
        "ddi_renal_double_hit": clean["nsaid"] * clean["acei_arb"],
        "ddi_renal_triple_whammy": (
            clean["nsaid"] * clean["acei_arb"] * clean["diuretic"]
        ),
        "ddi_cns_depression_synergy": clean["opioid"] * clean["benzodiazepine"],
        "ddi_bleeding_dual": clean["anticoagulant"] * clean["antiplatelet"],
        "ddi_bleeding_triple": (
            clean["nsaid"] * clean["anticoagulant"] * clean["antiplatelet"]
        ),
        "ddi_serotonergic_synergy": clean["ssri"] * clean["snri"],
        "ddi_qt_multidrug_load": clean[
            ["qt_prolonging_drug", "ssri", "snri"]
        ].sum(axis=1),
        "ddi_hepatic_triple_hit": (
            clean["hepatotoxic_drug_A"]
            * clean["antibiotic_hepatic_risk"]
            * clean["valproate_like_drug"]
        ),
    }
    for column, values in expected.items():
        checks[f"{column}_exact"] = bool(
            np.array_equal(clean[column].to_numpy(), np.asarray(values))
        )
        checks[f"{column}_positive_count"] = int(clean[column].sum())

    expected_burden = 0.08 * clean["active_drug_count"].to_numpy(dtype=float)
    for component, weight in ADR_BURDEN_COMPONENTS.items():
        expected_burden += weight * clean[component].to_numpy(dtype=float)
    expected_burden = np.round(expected_burden, 2)
    checks["cumulative_adr_burden_exact"] = bool(
        np.allclose(clean["cumulative_adr_burden"], expected_burden)
    )
    checks["severe_adr_burden_exact"] = bool(
        np.array_equal(
            clean["severe_adr_burden"].to_numpy(),
            (expected_burden >= 4.0).astype(int),
        )
    )

    hidden = samples["hidden_confounder"]
    checks["hidden_confounder_omits_latent"] = "unobserved_severity" not in hidden
    common = [column for column in hidden if column in clean]
    checks["hidden_matches_clean_world"] = bool(
        clean[common].equals(hidden[common])
    )
    selection = samples["selection_bias"]
    selected_clean = clean.loc[
        clean["hospital_contact"].eq(1)
        | clean["creatinine_tested"].eq(1)
        | clean["lft_tested"].eq(1)
    ]
    checks["selection_is_exact_subset"] = bool(
        selection.reset_index(drop=True).equals(selected_clean.reset_index(drop=True))
    )
    checks["noisy_has_documentation_columns"] = bool(
        samples["noisy_documentation"].shape[1] > clean.shape[1]
    )
    checks["multihospital_has_four_hospitals"] = (
        samples["multihospital"]["hospital_id"].nunique() == 4
    )
    for endpoint in ENDPOINTS:
        rate = float(clean[endpoint].mean())
        checks[f"{endpoint}_rate"] = rate
        checks[f"{endpoint}_nondegenerate"] = bool(0.001 < rate < 0.40)

    required_true = [
        "is_dag",
        "cumulative_adr_burden_exact",
        "severe_adr_burden_exact",
        "hidden_confounder_omits_latent",
        "hidden_matches_clean_world",
        "selection_is_exact_subset",
        "noisy_has_documentation_columns",
        "multihospital_has_four_hospitals",
    ]
    required_true += [f"{column}_exact" for column in expected]
    required_true += [f"{endpoint}_nondegenerate" for endpoint in ENDPOINTS]
    failures = [name for name in required_true if not checks[name]]
    if checks["duplicate_nodes"] or checks["duplicate_edges"] or checks["missing_edge_nodes"]:
        failures.append("graph_integrity_counts")
    if checks["hospitalization_parent_count"] < 8:
        failures.append("hospitalization_parent_count")
    for column in expected:
        if checks[f"{column}_positive_count"] == 0:
            failures.append(f"{column}_has_no_positive_examples")
    checks["failures"] = failures
    checks["status"] = "PASS" if not failures else "FAIL"
    output = DATA / "v2_2_validation_report.json"
    output.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
