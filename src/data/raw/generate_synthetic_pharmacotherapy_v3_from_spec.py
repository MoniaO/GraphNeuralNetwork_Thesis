#!/usr/bin/env python3
"""Generate six 20,000-patient v3 scenarios from the audited v3 DAG."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "2 v3. Data" / "dataset_v3"
V22_CODE = PROJECT_ROOT / "2. Data" / "code"
sys.path.insert(0, str(V22_CODE))

from generate_synthetic_pharmacotherapy_v2_from_spec import (  # noqa: E402
    bernoulli,
    initialize_patient_context,
    logit,
    sigmoid,
)
from build_v3_interaction_spec import (  # noqa: E402
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
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-dir", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=DATA)
    parser.add_argument("--n", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def topological_order(nodes: pd.DataFrame, edges: pd.DataFrame) -> list[str]:
    graph = nx.from_pandas_edgelist(
        edges, "source", "target", create_using=nx.DiGraph
    )
    graph.add_nodes_from(nodes["node"])
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("v3 specification is not a DAG")
    return list(nx.lexicographical_topological_sort(graph))


def base_probability(node: str, metadata: dict[str, dict[str, object]]) -> float:
    value = metadata.get(node, {}).get("base_prevalence", np.nan)
    if pd.notna(value):
        return float(value)
    node_type = metadata.get(node, {}).get("node_type", "")
    return {
        "drug_exposure": 0.16,
        "mechanism": 0.08,
        "adr_or_intermediate_state": 0.05,
        "observation_or_selection": 0.35,
        "clinical_endpoint": 0.02,
        "patient_context": 0.10,
    }.get(str(node_type), 0.10)


def deterministic_node(node: str, frame: pd.DataFrame) -> np.ndarray | None:
    if node == "active_drug_count":
        return frame[[d for d in DRUGS if d in frame]].sum(axis=1).to_numpy(int)

    # Preserve and expand v2.2 deterministic interaction semantics.
    old_gates = {
        "ddi_renal_double_hit": ["nsaid", "acei_arb"],
        "ddi_renal_triple_whammy": ["nsaid", "acei_arb", "diuretic"],
        "ddi_cns_depression_synergy": ["opioid", "benzodiazepine"],
        "ddi_bleeding_dual": ["anticoagulant", "antiplatelet"],
        "ddi_bleeding_triple": ["nsaid", "anticoagulant", "antiplatelet"],
        "ddi_serotonergic_synergy": ["ssri", "snri"],
        "ddi_hepatic_triple_hit": [
            "hepatotoxic_drug_A", "antibiotic_hepatic_risk", "valproate_like_drug",
        ],
    }
    if node in old_gates:
        return frame[old_gates[node]].prod(axis=1).to_numpy(int)
    if node == "ddi_qt_multidrug_load":
        return frame[
            ["qt_prolonging_drug", "ssri", "snri", "macrolide", "azole_antifungal"]
        ].sum(axis=1).to_numpy(int)

    if node in LOAD_DEFINITIONS:
        return frame[LOAD_DEFINITIONS[node]].sum(axis=1).to_numpy(int)

    if node in GATE_DEFINITIONS:
        if node == "ddi_serotonergic_high_load":
            return (frame["serotonergic_load"] >= 2).to_numpy(int)
        components = GATE_DEFINITIONS[node]
        return frame[components].gt(0).all(axis=1).to_numpy(int)

    if node == "cumulative_adr_burden":
        burden = np.zeros(len(frame), dtype=float)
        for component, weight in ADR_BURDEN_COMPONENTS_V3.items():
            burden += weight * frame[component].to_numpy(float)
        burden += 0.08 * frame["active_drug_count"].to_numpy(float)
        return np.round(burden, 2)
    if node == "severe_adr_burden":
        return (frame["cumulative_adr_burden"] >= 4.0).to_numpy(int)
    return None


def generate_world(
    n: int,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    scenario: str,
    rng: np.random.Generator,
    n_hospitals: int,
) -> pd.DataFrame:
    names = nodes["node"].tolist()
    metadata = nodes.set_index("node").to_dict(orient="index")
    incoming = {name: [] for name in names}
    for row in edges.to_dict(orient="records"):
        incoming[row["target"]].append(row)

    frame = pd.DataFrame({"patient_id": np.arange(1, n + 1)})
    frame = initialize_patient_context(
        frame, rng, scenario=scenario, n_hospitals=n_hospitals
    )
    # v3 is an older, multimorbid polypharmacy cohort. This is treatment
    # propensity, not the deterministic active-drug count.
    frame["polypharmacy_burden"] = np.clip(
        rng.poisson(
            4.7
            + 1.2 * frame["frailty"]
            + 0.9 * frame["heart_failure"]
            + 0.7 * frame["ckd"]
            + 0.4 * frame["depression_history"]
        ),
        0,
        16,
    )
    already = set(frame.columns)
    hospital_multiplier = frame["hospital_id"].map(
        {1: 1.00, 2: 1.18, 3: 0.86, 4: 1.10}
    ).fillna(1.0)
    hospital_doc_shift = frame["hospital_id"].map(
        {1: 0.00, 2: 0.20, 3: -0.20, 4: 0.08}
    ).fillna(0.0)

    for name in topological_order(nodes, edges):
        if name in already or name in {"patient_id", "hospital_id"}:
            continue
        deterministic = deterministic_node(name, frame)
        if deterministic is not None:
            frame[name] = deterministic
            continue

        meta = metadata[name]
        eta = np.full(n, logit(base_probability(name, metadata)), dtype=float)
        for incoming_edge in incoming[name]:
            source = incoming_edge["source"]
            if source not in frame:
                raise RuntimeError(
                    f"Parent {source} unavailable while generating {name}"
                )
            effect = (
                float(incoming_edge["effect_size"])
                if pd.notna(incoming_edge["effect_size"])
                else 0.5
            )
            sign = (
                float(incoming_edge["effect_sign"])
                if pd.notna(incoming_edge["effect_sign"])
                else 1.0
            )
            values = frame[source].to_numpy(float)
            finite = values[np.isfinite(values)]
            unique = np.unique(finite)
            if len(unique) > 2:
                scale = float(np.std(finite))
                if scale > 1e-8:
                    values = (values - float(np.mean(finite))) / scale
            eta += sign * effect * values

        if meta["node_type"] == "drug_exposure":
            eta += np.log(hospital_multiplier.to_numpy(float))
        if meta["node_type"] == "observation_or_selection":
            eta += hospital_doc_shift.to_numpy(float)

        if scenario == "no_overlap":
            low_egfr = (frame["baseline_egfr"] < 25).to_numpy(float)
            high_burden = (frame["polypharmacy_burden"] >= 8).to_numpy(float)
            if name in {"nsaid", "metformin", "lithium"}:
                eta -= 3.0 * low_egfr
            if name == "acei_arb":
                eta -= 1.3 * low_egfr
            if name in {"anticoagulant", "antiplatelet"}:
                eta += 1.5 * high_burden

        frame[name] = bernoulli(rng, sigmoid(eta))

    # Recorded variables are deterministic products in every world.
    for state, tested, recorded in [
        ("creatinine_rise", "creatinine_tested", "elevated_creatinine_recorded"),
        ("alt_ast_rise", "lft_tested", "elevated_alt_ast_recorded"),
        ("qt_prolongation_state", "ecg_performed", "qt_recorded"),
    ]:
        if {state, tested}.issubset(frame):
            frame[recorded] = (frame[state] & frame[tested]).astype(int)

    missing = [name for name in names if name not in frame]
    if missing:
        raise RuntimeError(f"Declared nodes not generated: {missing}")
    ordered = ["patient_id", "hospital_id"] + [
        name for name in names if name not in {"patient_id", "hospital_id"}
    ]
    extras = [column for column in frame.columns if column not in ordered]
    return frame[ordered + extras]


def make_noisy_documentation(
    frame: pd.DataFrame, nodes: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    output = frame.copy()
    observability = nodes.set_index("node")["observability"].to_dict()
    states = ENDPOINTS + [
        "creatinine_rise", "alt_ast_rise", "confusion_state", "dizziness",
        "mood_lowering", "hyponatremia_state", "hyperkalemia_state",
        "qt_prolongation_state", "muscle_injury", "lactate_accumulation",
        "serotonin_toxicity", "digoxin_toxicity", "lithium_toxicity",
    ]
    for column in [c for c in states if c in output]:
        truth = output[column].to_numpy(int)
        sensitivity = observability.get(column, 0.70)
        sensitivity = 0.70 if pd.isna(sensitivity) else float(sensitivity)
        recorded = np.where(
            truth == 1,
            rng.binomial(1, sensitivity, len(output)),
            rng.binomial(1, 0.01, len(output)),
        )
        output[f"true_{column}"] = truth
        output[f"recorded_{column}"] = recorded.astype(int)
    return output


def summary_row(scenario: str, frame: pd.DataFrame) -> dict[str, object]:
    row: dict[str, object] = {
        "scenario": scenario,
        "n_rows": len(frame),
        "n_columns": frame.shape[1],
    }
    for endpoint in ENDPOINTS:
        row[f"{endpoint}_rate"] = float(frame[endpoint].mean())
    for column in [
        *LOAD_DEFINITIONS,
        *GATE_DEFINITIONS,
        "severe_adr_burden",
    ]:
        row[f"{column}_mean_or_rate"] = float(frame[column].mean())
    row["active_drug_count_mean"] = float(frame["active_drug_count"].mean())
    row["active_drug_count_p05"] = float(frame["active_drug_count"].quantile(0.05))
    row["active_drug_count_p95"] = float(frame["active_drug_count"].quantile(0.95))
    row["cumulative_adr_burden_mean"] = float(
        frame["cumulative_adr_burden"].mean()
    )
    return row


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    nodes = pd.read_csv(args.spec_dir / "synthetic_pharmacotherapy_v3_nodes.csv")
    edges = pd.read_csv(
        args.spec_dir / "synthetic_pharmacotherapy_v3_edges_audited.csv"
    )
    rng = np.random.default_rng(args.seed)

    clean = generate_world(args.n, nodes, edges, "clean", rng, 1)
    hidden = clean.drop(columns=["unobserved_severity"])
    selection_mask = (
        clean["hospital_contact"].eq(1)
        | clean["creatinine_tested"].eq(1)
        | clean["lft_tested"].eq(1)
    )
    selection = clean.loc[selection_mask].copy()
    no_overlap = generate_world(args.n, nodes, edges, "no_overlap", rng, 1)
    noisy = make_noisy_documentation(clean, nodes, rng)
    multihospital = generate_world(
        args.n, nodes, edges, "multihospital", rng, 4
    )
    datasets = {
        "clean": clean,
        "hidden_confounder": hidden,
        "selection_bias": selection,
        "no_overlap": no_overlap,
        "noisy_documentation": noisy,
        "multihospital": multihospital,
    }
    for scenario, frame in datasets.items():
        frame.to_csv(
            args.output_dir
            / f"synthetic_pharmacotherapy_v3_samples_{scenario}.csv",
            index=False,
        )

    summary = pd.DataFrame(
        [summary_row(scenario, frame) for scenario, frame in datasets.items()]
    )
    summary.to_csv(
        args.output_dir / "synthetic_pharmacotherapy_v3_dataset_summary.csv",
        index=False,
    )
    config = {
        "version": "v3",
        "seed": args.seed,
        "n_requested_per_full_scenario": args.n,
        "target_mean_active_drugs": 6.0,
        "graph_truth": "synthetic_pharmacotherapy_v3_edges_audited.csv",
        "interaction_gates": True,
        "graded_multidrug_loads": True,
        "drug_disease_interactions": True,
        "equifinal_adr_routes": True,
        "cumulative_adr_burden": True,
        "scenarios": SCENARIOS,
    }
    (args.output_dir / "v3_generation_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
