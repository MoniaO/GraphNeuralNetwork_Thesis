#!/usr/bin/env python3
"""Build the expanded, interaction-rich v3 pharmacotherapy DAG.

v3 extends (but never overwrites) the audited v2.2 graph. It adds drug classes,
diseases, drug-drug and drug-disease gates, graded multi-drug loads, independent
mechanistic routes to shared ADRs, and three rare clinical endpoints.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V22_DIR = PROJECT_ROOT / "2. Data" / "dataset_v2_2"
OUTPUT_DIR = PROJECT_ROOT / "2 v3. Data" / "dataset_v3"

V22_CODE = PROJECT_ROOT / "2. Data" / "code"
sys.path.insert(0, str(V22_CODE))
from build_v2_2_interaction_spec import DRUGS as V22_DRUGS  # noqa: E402

NEW_DRUGS = [
    "statin",
    "macrolide",
    "azole_antifungal",
    "metformin",
    "digoxin",
    "triptan",
    "lithium",
    "potassium_supplement",
]
DRUGS = V22_DRUGS + NEW_DRUGS

NEW_DISEASES = [
    "hypertension",
    "atrial_fibrillation",
    "epilepsy",
    "hypothyroidism",
    "obesity",
]

NEW_ENDPOINTS = ["Serotonin_syndrome", "Rhabdomyolysis", "Lactic_acidosis"]
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
] + NEW_ENDPOINTS

LOAD_DEFINITIONS = {
    "nephrotoxin_load": [
        "nsaid", "aminoglycoside", "contrast_agent", "lithium", "acei_arb",
    ],
    "hepatic_drug_load": [
        "hepatotoxic_drug_A", "antibiotic_hepatic_risk", "valproate_like_drug",
        "statin", "macrolide", "azole_antifungal",
    ],
    "qt_drug_load_v3": [
        "qt_prolonging_drug", "ssri", "snri", "macrolide",
        "azole_antifungal", "digoxin",
    ],
    "cns_depressant_load": [
        "opioid", "benzodiazepine", "anticholinergic", "valproate_like_drug",
    ],
    "bleeding_risk_load": [
        "anticoagulant", "antiplatelet", "nsaid", "ssri", "corticosteroid",
    ],
    "serotonergic_load": ["ssri", "snri", "opioid", "triptan"],
    "cyp_inhibitor_load": ["macrolide", "azole_antifungal"],
}

GATE_DEFINITIONS = {
    "ddi_statin_cyp_inhibitor": ["statin", "cyp_inhibitor_load"],
    "ddi_metformin_renal_risk": ["metformin", "ckd"],
    "ddi_lithium_renal_risk": ["lithium", "ckd"],
    "ddi_digoxin_electrolyte_risk": ["digoxin", "electrolyte_disturbance"],
    "ddi_serotonergic_high_load": ["serotonergic_load"],
    "drug_disease_nsaid_ckd": ["nsaid", "ckd"],
    "drug_disease_nsaid_heart_failure": ["nsaid", "heart_failure"],
    "drug_disease_qt_baseline_risk": ["qt_drug_load_v3", "baseline_qt_risk"],
    "drug_disease_hepatic_liver_disease": ["hepatic_drug_load", "liver_disease"],
    "drug_disease_anticoagulant_frailty": ["anticoagulant", "frailty"],
}

ADR_BURDEN_COMPONENTS_V3 = {
    "creatinine_rise": 2.0,
    "alt_ast_rise": 1.5,
    "bilirubin_rise": 2.0,
    "overt_bleeding": 3.0,
    "confusion_state": 1.5,
    "sedation_state": 1.0,
    "qt_prolongation_state": 2.0,
    "hyponatremia_state": 1.5,
    "hyperkalemia_state": 2.0,
    "hypotension": 1.0,
    "dizziness": 0.5,
    "muscle_injury": 2.0,
    "lactate_accumulation": 2.5,
    "serotonin_toxicity": 2.0,
    "digoxin_toxicity": 2.0,
    "lithium_toxicity": 2.0,
}


def node(
    name: str,
    node_type: str,
    layer: str,
    description: str,
    prevalence: float,
    severity: float,
    observability: float,
    *,
    value_type: str = "binary",
    gate: str = "",
    endpoint: bool = False,
) -> dict[str, object]:
    rarity = float(np.clip(1.0 / np.sqrt(max(prevalence, 1e-4)), 1.0, 10.0))
    if prevalence < 0.01:
        frequency = "very_rare"
    elif prevalence < 0.03:
        frequency = "rare"
    elif prevalence < 0.10:
        frequency = "occasional"
    else:
        frequency = "common"
    return {
        "node": name,
        "node_type": node_type,
        "layer": layer,
        "description": description,
        "base_prevalence": prevalence,
        "frequency_class": frequency,
        "severity_weight": severity,
        "observability": observability,
        "rarity_weight": round(rarity, 3),
        "node_priority_weight": round(min(10.0, severity * rarity), 3),
        "is_endpoint": endpoint,
        "is_rare_signal": prevalence < 0.03,
        "is_latent": False,
        "recommended_use": "target" if endpoint else "feature",
        "value_type": value_type,
        "interaction_gate": gate,
        "introduced_in": "v3",
    }


def new_nodes() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    drug_specs = {
        "statin": ("Statin exposure", 0.30),
        "macrolide": ("Macrolide antibiotic exposure", 0.13),
        "azole_antifungal": ("Azole antifungal exposure", 0.08),
        "metformin": ("Metformin exposure", 0.22),
        "digoxin": ("Digoxin exposure", 0.08),
        "triptan": ("Triptan exposure", 0.09),
        "lithium": ("Lithium exposure", 0.05),
        "potassium_supplement": ("Potassium supplement exposure", 0.12),
    }
    for name, (description, prevalence) in drug_specs.items():
        rows.append(node(
            name, "drug_exposure", "2_drugs", description,
            prevalence, 2.5, 0.85,
        ))

    disease_specs = {
        "hypertension": ("Hypertension", 0.48),
        "atrial_fibrillation": ("Atrial fibrillation", 0.12),
        "epilepsy": ("Epilepsy", 0.035),
        "hypothyroidism": ("Hypothyroidism", 0.11),
        "obesity": ("Obesity", 0.31),
    }
    for name, (description, prevalence) in disease_specs.items():
        rows.append(node(
            name, "patient_context", "1_patient_context", description,
            prevalence, 2.0, 0.90,
        ))

    for name, components in LOAD_DEFINITIONS.items():
        rows.append(node(
            name, "mechanism", "3_mechanisms",
            f"Observed count across: {', '.join(components)}",
            0.20, 3.0, 1.0, value_type="count",
            gate="sum:" + "&".join(components),
        ))
    for name, components in GATE_DEFINITIONS.items():
        rows.append(node(
            name, "mechanism", "3_mechanisms",
            f"Deterministic interaction gate: {' AND '.join(components)}",
            0.02, 4.0, 0.9, gate="and:" + "&".join(components),
        ))

    for name, description, prevalence in [
        ("muscle_injury", "Drug-associated skeletal muscle injury", 0.025),
        ("lactate_accumulation", "Pathological lactate accumulation", 0.012),
        ("serotonin_toxicity", "Excess serotonergic activity", 0.012),
        ("digoxin_toxicity", "Digoxin toxicity state", 0.010),
        ("lithium_toxicity", "Lithium toxicity state", 0.008),
    ]:
        rows.append(node(
            name, "adr_or_intermediate_state", "4_intermediate_states",
            description, prevalence, 4.0, 0.75,
        ))

    for name, description, prevalence in [
        ("Serotonin_syndrome", "Clinical serotonin syndrome", 0.0025),
        ("Rhabdomyolysis", "Clinical rhabdomyolysis", 0.0035),
        ("Lactic_acidosis", "Clinical lactic acidosis", 0.0040),
    ]:
        rows.append(node(
            name, "clinical_endpoint", "6_endpoints", description,
            prevalence, 5.0, 0.70, endpoint=True,
        ))
    return rows


def edge(
    source: str,
    target: str,
    edge_type: str,
    effect: float,
    pathway: str,
    rationale: str,
    sign: int = 1,
) -> dict[str, object]:
    return {
        "source": source,
        "target": target,
        "edge_type": edge_type,
        "effect_sign": sign,
        "effect_size": effect,
        "activation_frequency": 0.8,
        "mechanistic_confidence": 0.80,
        "evidence_weight": 0.75,
        "temporal_lag": "same_episode",
        "dag_motif": "interaction_gate" if "gate" in edge_type else "mediator",
        "clinical_pathway": pathway,
        "hcr_conditioning_hint": "polypharmacy_burden,unobserved_severity",
        "hcr_information_weight": np.nan,
        "gnn_edge_weight": np.nan,
        "rationale": rationale,
        "introduced_in": "v3",
    }


def new_edges() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    # Demographics and comorbidities.
    context_edges = [
        ("age", "hypertension", 0.65), ("age", "atrial_fibrillation", 0.70),
        ("age", "hypothyroidism", 0.25), ("sex_female", "hypothyroidism", 0.60),
        ("diabetes", "obesity", 0.60), ("obesity", "hypertension", 0.70),
        ("hypertension", "heart_failure", 0.55),
        ("hypertension", "ckd", 0.35),
        ("age", "Falls", 0.30), ("age", "Delirium", 0.25),
        ("age", "AKI", 0.25), ("sex_female", "QT_arrhythmia", 0.30),
        ("sex_female", "DILI", 0.18), ("sex_female", "Hyponatremia", 0.25),
    ]
    for source, target, effect in context_edges:
        rows.append(edge(
            source, target, "risk_modifier", effect, "demographic_context",
            "Explicit demographic or comorbidity risk modifier.",
        ))

    # Clinically plausible treatment assignment.
    treatment_edges = [
        ("obesity", "metformin", 0.45), ("diabetes", "metformin", 1.10),
        ("hypertension", "acei_arb", 0.75), ("hypertension", "diuretic", 0.45),
        ("atrial_fibrillation", "anticoagulant", 1.15),
        ("atrial_fibrillation", "digoxin", 0.85),
        ("epilepsy", "valproate_like_drug", 1.00),
        ("hypothyroidism", "statin", 0.30),
        ("depression_history", "lithium", 0.35),
        ("baseline_potassium", "potassium_supplement", -0.55),
    ]
    for source, target, effect in treatment_edges:
        sign = -1 if effect < 0 else 1
        rows.append(edge(
            source, target, "treatment_indication", abs(effect),
            "treatment_assignment", "Context influences treatment assignment.", sign,
        ))

    # Every new drug contributes to observed active-drug count.
    for drug in NEW_DRUGS:
        rows.append(edge(
            drug, "active_drug_count", "drug_to_burden", 1.0,
            "polypharmacy", "Deterministic contribution to active-drug count.",
        ))

    # Drug components feed graded load nodes.
    for load, components in LOAD_DEFINITIONS.items():
        for component in components:
            rows.append(edge(
                component, load, "drug_to_load", 1.0, "multidrug_load",
                f"Component of deterministic {load}.",
            ))

    # Required components of deterministic drug-drug/drug-disease gates.
    for gate, components in GATE_DEFINITIONS.items():
        for component in components:
            rows.append(edge(
                component, gate, "interaction_gate_component", 1.0,
                "drug_interaction", f"Required component of {gate}.",
            ))

    # Multiple independent routes to shared ADRs/endpoints.
    effects = [
        ("nephrotoxin_load", "nephrotoxic_stress", 0.70, "renal"),
        ("nephrotoxin_load", "tubular_injury", 0.60, "renal"),
        ("drug_disease_nsaid_ckd", "creatinine_rise", 1.00, "renal"),
        ("drug_disease_nsaid_heart_failure", "reduced_renal_perfusion", 0.90, "renal"),
        ("ddi_lithium_renal_risk", "lithium_toxicity", 1.25, "renal"),
        ("lithium_toxicity", "confusion_state", 0.85, "cns"),
        ("hepatic_drug_load", "hepatic_metabolic_stress", 0.65, "hepatic"),
        ("hepatic_drug_load", "cholestatic_pattern", 0.45, "hepatic"),
        ("drug_disease_hepatic_liver_disease", "alt_ast_rise", 1.00, "hepatic"),
        ("cyp_inhibitor_load", "hepatic_metabolic_stress", 0.55, "hepatic"),
        ("qt_drug_load_v3", "cardiac_repolarization_delay", 0.65, "qt"),
        ("drug_disease_qt_baseline_risk", "qt_prolongation_state", 1.00, "qt"),
        ("ddi_digoxin_electrolyte_risk", "digoxin_toxicity", 1.15, "cardiac"),
        ("digoxin_toxicity", "QT_arrhythmia", 0.75, "cardiac"),
        ("cns_depressant_load", "cns_sedation", 0.65, "cns"),
        ("cns_depressant_load", "psychomotor_impairment", 0.55, "cns"),
        ("bleeding_risk_load", "coagulation_impairment", 0.60, "bleeding"),
        ("drug_disease_anticoagulant_frailty", "overt_bleeding", 0.85, "bleeding"),
        ("serotonergic_load", "serotonergic_shift", 0.65, "serotonergic"),
        ("ddi_serotonergic_high_load", "serotonin_toxicity", 1.10, "serotonergic"),
        ("serotonin_toxicity", "Serotonin_syndrome", 1.65, "serotonergic"),
        ("ddi_statin_cyp_inhibitor", "muscle_injury", 1.50, "muscle"),
        ("hypothyroidism", "muscle_injury", 0.45, "muscle"),
        ("muscle_injury", "Rhabdomyolysis", 1.75, "muscle"),
        ("ddi_metformin_renal_risk", "lactate_accumulation", 1.35, "metabolic"),
        ("metformin", "lactate_accumulation", 0.25, "metabolic"),
        ("lactate_accumulation", "Lactic_acidosis", 1.75, "metabolic"),
    ]
    for source, target, effect, pathway in effects:
        rows.append(edge(
            source, target, "interaction_amplify", effect, pathway,
            "v3 independent or interaction-amplified mechanistic route.",
        ))

    for component in [
        "muscle_injury", "lactate_accumulation", "serotonin_toxicity",
        "digoxin_toxicity", "lithium_toxicity",
    ]:
        rows.append(edge(
            component, "cumulative_adr_burden", "adr_burden_component",
            ADR_BURDEN_COMPONENTS_V3[component], "cumulative_adr",
            "Severity-weighted component of v3 cumulative ADR burden.",
        ))
    for endpoint, effect_size in [
        ("Serotonin_syndrome", 1.25),
        ("Rhabdomyolysis", 1.25),
        ("Lactic_acidosis", 1.35),
    ]:
        rows.append(edge(
            endpoint, "Hospitalization", "endpoint_to_hospitalization",
            effect_size, "hospitalization",
            f"{endpoint} contributes to hospitalization.",
        ))
    return rows


def make_adjacency(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    names = nodes["node"].tolist()
    adjacency = pd.DataFrame(0, index=names, columns=names, dtype=np.int8)
    for source, target in edges[["source", "target"]].itertuples(index=False, name=None):
        adjacency.loc[source, target] = 1
    adjacency.index.name = "source"
    return adjacency.reset_index()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nodes = pd.read_csv(V22_DIR / "synthetic_pharmacotherapy_v2_2_nodes.csv")
    edges = pd.read_csv(V22_DIR / "synthetic_pharmacotherapy_v2_2_edges_audited.csv")

    # Increase exposure prevalence to target ~6 active drugs per patient.
    nodes.loc[nodes["node_type"].eq("drug_exposure"), "base_prevalence"] *= 2.15
    nodes.loc[nodes["node_type"].eq("drug_exposure"), "base_prevalence"] = (
        nodes.loc[nodes["node_type"].eq("drug_exposure"), "base_prevalence"]
        .clip(upper=0.45)
    )
    added_nodes = pd.DataFrame(new_nodes()).reindex(columns=nodes.columns)
    nodes = pd.concat([nodes, added_nodes], ignore_index=True)

    proposed_edges = pd.DataFrame(new_edges())
    existing_pairs = set(map(tuple, edges[["source", "target"]].to_numpy()))
    proposed_edges = proposed_edges[
        ~proposed_edges[["source", "target"]].apply(tuple, axis=1).isin(existing_pairs)
    ].copy()
    start = int(edges["edge_id"].str.extract(r"(\d+)")[0].astype(int).max()) + 1
    proposed_edges.insert(
        0, "edge_id",
        [f"e{i:03d}" for i in range(start, start + len(proposed_edges))],
    )
    proposed_edges = proposed_edges.reindex(columns=edges.columns)
    all_edges = pd.concat([edges, proposed_edges], ignore_index=True)

    missing = (
        set(all_edges["source"]) | set(all_edges["target"])
    ).difference(nodes["node"])
    if missing:
        raise ValueError(f"Edges reference missing nodes: {sorted(missing)}")
    if all_edges.duplicated(["source", "target"]).any():
        raise ValueError("Duplicate source-target pairs in v3 graph.")
    graph = nx.from_pandas_edgelist(
        all_edges, "source", "target", create_using=nx.DiGraph
    )
    graph.add_nodes_from(nodes["node"])
    if not nx.is_directed_acyclic_graph(graph):
        cycle = nx.find_cycle(graph)
        raise ValueError(f"v3 graph is not a DAG: {cycle}")

    nodes.to_csv(OUTPUT_DIR / "synthetic_pharmacotherapy_v3_nodes.csv", index=False)
    all_edges.to_csv(
        OUTPUT_DIR / "synthetic_pharmacotherapy_v3_edges_audited.csv", index=False
    )
    proposed_edges.to_csv(
        OUTPUT_DIR / "synthetic_pharmacotherapy_v3_added_edges_with_rationale.csv",
        index=False,
    )
    make_adjacency(nodes, all_edges).to_csv(
        OUTPUT_DIR / "synthetic_pharmacotherapy_v3_adjacency_audited.csv",
        index=False,
    )
    manifest = {
        "graph_version": "v3_audited",
        "base_graph": "v2.2_audited",
        "nodes": len(nodes),
        "edges": len(all_edges),
        "added_nodes": len(added_nodes),
        "added_edges": len(proposed_edges),
        "is_dag": True,
        "drugs": DRUGS,
        "new_diseases": NEW_DISEASES,
        "endpoints": ENDPOINTS,
        "load_definitions": LOAD_DEFINITIONS,
        "gate_definitions": GATE_DEFINITIONS,
        "adr_burden_components": ADR_BURDEN_COMPONENTS_V3,
    }
    (OUTPUT_DIR / "v3_graph_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
