#!/usr/bin/env python3
"""Validate v3 graph integrity, deterministic semantics, scenarios, and rates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE = PROJECT_ROOT / "2 v3. Data" / "code"
DATA = PROJECT_ROOT / "2 v3. Data" / "dataset_v3"
sys.path.insert(0, str(CODE))

from build_v3_interaction_spec import (  # pyright: ignore[reportMissingImports]  # noqa: E402
    ADR_BURDEN_COMPONENTS_V3,
    DRUGS,
    ENDPOINTS,
    GATE_DEFINITIONS,
    LOAD_DEFINITIONS,
)

SCENARIOS = [
    "clean",
    "hidden_confounder",
    "selection_bias",
    "no_overlap",
    "noisy_documentation",
    "multihospital",
]

EXPECTED_ISOLATED_NODES = {"cognitive_slowing"}


def exact_gate(name: str, frame: pd.DataFrame) -> np.ndarray:
    if name == "ddi_serotonergic_high_load":
        return (frame["serotonergic_load"] >= 2).to_numpy(int)
    return frame[GATE_DEFINITIONS[name]].gt(0).all(axis=1).to_numpy(int)


def main() -> None:
    nodes = pd.read_csv(DATA / "synthetic_pharmacotherapy_v3_nodes.csv")
    edges = pd.read_csv(DATA / "synthetic_pharmacotherapy_v3_edges_audited.csv")
    clean = pd.read_csv(DATA / "synthetic_pharmacotherapy_v3_samples_clean.csv")

    graph = nx.from_pandas_edgelist(
        edges, "source", "target", create_using=nx.DiGraph
    )
    graph.add_nodes_from(nodes["node"])
    isolated_nodes = sorted(nx.isolates(graph))
    unexpected_isolated_nodes = sorted(
        set(isolated_nodes).difference(EXPECTED_ISOLATED_NODES)
    )
    source_only_nodes = sorted(
        node
        for node in graph
        if graph.in_degree(node) == 0 and graph.out_degree(node) > 0
    )
    sink_only_nodes = sorted(
        node
        for node in graph
        if graph.in_degree(node) > 0 and graph.out_degree(node) == 0
    )
    checks: dict[str, object] = {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "is_dag": nx.is_directed_acyclic_graph(graph),
        "duplicate_nodes": int(nodes["node"].duplicated().sum()),
        "duplicate_edges": int(edges[["source", "target"]].duplicated().sum()),
        "missing_edge_nodes": len(
            (set(edges["source"]) | set(edges["target"])).difference(nodes["node"])
        ),
        "isolated_node_count": len(isolated_nodes),
        "isolated_nodes": isolated_nodes,
        "expected_isolated_nodes": sorted(EXPECTED_ISOLATED_NODES),
        "unexpected_isolated_nodes": unexpected_isolated_nodes,
        "source_only_node_count": len(source_only_nodes),
        "source_only_nodes": source_only_nodes,
        "sink_only_node_count": len(sink_only_nodes),
        "sink_only_nodes": sink_only_nodes,
        "clean_patient_count": len(clean),
        "active_drug_count_exact": bool(
            np.array_equal(
                clean["active_drug_count"].to_numpy(int),
                clean[DRUGS].sum(axis=1).to_numpy(int),
            )
        ),
        "active_drug_count_mean": float(clean["active_drug_count"].mean()),
        "active_drug_count_p05": float(clean["active_drug_count"].quantile(0.05)),
        "active_drug_count_p95": float(clean["active_drug_count"].quantile(0.95)),
    }

    for name, components in LOAD_DEFINITIONS.items():
        checks[f"{name}_exact"] = bool(
            np.array_equal(
                clean[name].to_numpy(int),
                clean[components].sum(axis=1).to_numpy(int),
            )
        )
        checks[f"{name}_positive_count"] = int(clean[name].gt(0).sum())
    for name in GATE_DEFINITIONS:
        checks[f"{name}_exact"] = bool(
            np.array_equal(clean[name].to_numpy(int), exact_gate(name, clean))
        )
        checks[f"{name}_positive_count"] = int(clean[name].sum())

    expected_burden = np.zeros(len(clean), dtype=float)
    for component, weight in ADR_BURDEN_COMPONENTS_V3.items():
        expected_burden += weight * clean[component].to_numpy(float)
    expected_burden += 0.08 * clean["active_drug_count"].to_numpy(float)
    checks["cumulative_adr_burden_exact"] = bool(
        np.allclose(clean["cumulative_adr_burden"], np.round(expected_burden, 2))
    )
    checks["severe_adr_burden_exact"] = bool(
        np.array_equal(
            clean["severe_adr_burden"].to_numpy(int),
            (clean["cumulative_adr_burden"] >= 4.0).to_numpy(int),
        )
    )

    hidden = pd.read_csv(
        DATA / "synthetic_pharmacotherapy_v3_samples_hidden_confounder.csv"
    )
    selection = pd.read_csv(
        DATA / "synthetic_pharmacotherapy_v3_samples_selection_bias.csv"
    )
    noisy = pd.read_csv(
        DATA / "synthetic_pharmacotherapy_v3_samples_noisy_documentation.csv"
    )
    multihospital = pd.read_csv(
        DATA / "synthetic_pharmacotherapy_v3_samples_multihospital.csv"
    )
    checks["hidden_confounder_omits_latent"] = "unobserved_severity" not in hidden
    checks["hidden_matches_clean_patient_ids"] = hidden["patient_id"].equals(
        clean["patient_id"]
    )
    checks["selection_is_exact_subset"] = set(selection["patient_id"]).issubset(
        set(clean["patient_id"])
    )
    checks["noisy_has_documentation_columns"] = all(
        f"recorded_{endpoint}" in noisy for endpoint in ENDPOINTS
    )
    checks["multihospital_has_four_hospitals"] = (
        multihospital["hospital_id"].nunique() == 4
    )

    rate_targets = {
        "DILI": (0.005, 0.04),
        "QT_arrhythmia": (0.005, 0.04),
        "Serotonin_syndrome": (0.0005, 0.02),
        "Rhabdomyolysis": (0.0005, 0.02),
        "Lactic_acidosis": (0.0005, 0.02),
        "AKI": (0.05, 0.25),
        "Falls": (0.03, 0.20),
        "Hospitalization": (0.10, 0.40),
    }
    for endpoint in ENDPOINTS:
        rate = float(clean[endpoint].mean())
        checks[f"{endpoint}_rate"] = rate
        checks[f"{endpoint}_nondegenerate"] = bool(0 < clean[endpoint].sum() < len(clean))
        if endpoint in rate_targets:
            lower, upper = rate_targets[endpoint]
            checks[f"{endpoint}_within_target"] = bool(lower <= rate <= upper)

    required_true = [
        "is_dag",
        "active_drug_count_exact",
        "cumulative_adr_burden_exact",
        "severe_adr_burden_exact",
        "hidden_confounder_omits_latent",
        "hidden_matches_clean_patient_ids",
        "selection_is_exact_subset",
        "noisy_has_documentation_columns",
        "multihospital_has_four_hospitals",
        *[f"{name}_exact" for name in LOAD_DEFINITIONS],
        *[f"{name}_exact" for name in GATE_DEFINITIONS],
        *[f"{endpoint}_nondegenerate" for endpoint in ENDPOINTS],
        *[f"{endpoint}_within_target" for endpoint in rate_targets],
    ]
    failures = [name for name in required_true if checks.get(name) is not True]
    if checks["duplicate_nodes"]:
        failures.append("duplicate_nodes")
    if checks["duplicate_edges"]:
        failures.append("duplicate_edges")
    if checks["missing_edge_nodes"]:
        failures.append("missing_edge_nodes")
    if checks["unexpected_isolated_nodes"]:
        failures.append("unexpected_isolated_nodes")
    if not 5.5 <= checks["active_drug_count_mean"] <= 6.5:
        failures.append("active_drug_count_mean_outside_5.5_to_6.5")

    checks["failures"] = failures
    checks["status"] = "PASS" if not failures else "FAIL"
    report = DATA / "v3_validation_report.json"
    report.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
