"""Truth-graph motif catalog for Wave 3B higher-order HCR experiments.

Sources of truth (frozen):
  - audited edges/nodes CSVs under dataset_v3
  - GATE_DEFINITIONS / LOAD_DEFINITIONS / old_gates in the v3 generator
    (copied here so motif extraction does not import the full generator).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

# Frozen copies of generator definitions (do not drift without a catalog re-export).
LOAD_DEFINITIONS: dict[str, list[str]] = {
    "nephrotoxin_load": ["nsaid", "aminoglycoside", "contrast_agent", "lithium", "acei_arb"],
    "hepatic_drug_load": [
        "hepatotoxic_drug_A",
        "antibiotic_hepatic_risk",
        "valproate_like_drug",
        "statin",
        "macrolide",
        "azole_antifungal",
    ],
    "qt_drug_load_v3": [
        "qt_prolonging_drug",
        "ssri",
        "snri",
        "macrolide",
        "azole_antifungal",
        "digoxin",
    ],
    "cns_depressant_load": ["opioid", "benzodiazepine", "anticholinergic", "valproate_like_drug"],
    "bleeding_risk_load": ["anticoagulant", "antiplatelet", "nsaid", "ssri", "corticosteroid"],
    "serotonergic_load": ["ssri", "snri", "opioid", "triptan"],
    "cyp_inhibitor_load": ["macrolide", "azole_antifungal"],
}

GATE_DEFINITIONS: dict[str, list[str]] = {
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

OLD_GATES: dict[str, list[str]] = {
    "ddi_renal_double_hit": ["nsaid", "acei_arb"],
    "ddi_renal_triple_whammy": ["nsaid", "acei_arb", "diuretic"],
    "ddi_cns_depression_synergy": ["opioid", "benzodiazepine"],
    "ddi_bleeding_dual": ["anticoagulant", "antiplatelet"],
    "ddi_bleeding_triple": ["nsaid", "anticoagulant", "antiplatelet"],
    "ddi_serotonergic_synergy": ["ssri", "snri"],
    "ddi_hepatic_triple_hit": [
        "hepatotoxic_drug_A",
        "antibiotic_hepatic_risk",
        "valproate_like_drug",
    ],
}


@dataclass(frozen=True)
class DualGate:
    gate: str
    parent_a: str
    parent_b: str
    children: tuple[str, ...]
    source: str  # gate_definitions | old_gates | edges
    binary_parents: bool


@dataclass(frozen=True)
class TripleGate:
    gate: str
    parent_a: str
    parent_b: str
    parent_c: str
    children: tuple[str, ...]
    source: str
    binary_parents: bool


@dataclass(frozen=True)
class LoadNode:
    load: str
    components: tuple[str, ...]


@dataclass(frozen=True)
class ConfoundedNonEdge:
    x: str
    y: str
    common_cause: str


@dataclass(frozen=True)
class MotifCompletionTask:
    """Hide one parent→gate edge; co-parent is the oracle Z for HCR-3."""

    gate: str
    hidden_parent: str
    visible_parent: str  # Z / co-parent
    candidate_source: str  # = hidden_parent
    candidate_target: str  # = gate
    children: tuple[str, ...]


def _resolve_paths(cfg_or_root: Any = None) -> tuple[Path, Path]:
    if cfg_or_root is None:
        import os

        root = Path(
            os.environ.get(
                "GSN_PROJECT_ROOT",
                str(Path.home() / "Desktop" / "GSN Graphs dysertation 2026"),
            )
        ).expanduser() / "2 v3. Data" / "dataset_v3"
    elif hasattr(cfg_or_root, "data"):
        root = Path(str(cfg_or_root.data.dataset.root_dir)).expanduser().resolve()
    else:
        root = Path(cfg_or_root).expanduser().resolve()
    return root / "synthetic_pharmacotherapy_v3_nodes.csv", root / "synthetic_pharmacotherapy_v3_edges_audited.csv"


def load_truth_graph(cfg_or_root: Any = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes_path, edges_path = _resolve_paths(cfg_or_root)
    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)
    return nodes, edges


def _is_binary_column(samples: pd.DataFrame | None, name: str) -> bool:
    if "_load" in name or name.endswith("_count") or name.endswith("_burden"):
        return False
    if samples is None or name not in samples.columns:
        return True
    vals = set(pd.unique(samples[name].dropna().astype(float).round(10)))
    return vals.issubset({0.0, 1.0})


def _children_of(edges: pd.DataFrame, gate: str) -> tuple[str, ...]:
    outs = edges.loc[edges["source"].astype(str) == gate, "target"].astype(str).tolist()
    return tuple(sorted(set(outs)))


def _all_gate_definitions() -> dict[str, list[str]]:
    merged = dict(OLD_GATES)
    merged.update(GATE_DEFINITIONS)
    return merged


def extract_dual_gates(
    edges: pd.DataFrame,
    samples: pd.DataFrame | None = None,
) -> list[DualGate]:
    duals: list[DualGate] = []
    for gate, comps in _all_gate_definitions().items():
        if len(comps) != 2:
            continue
        a, b = comps[0], comps[1]
        duals.append(
            DualGate(
                gate=gate,
                parent_a=a,
                parent_b=b,
                children=_children_of(edges, gate),
                source="old_gates" if gate in OLD_GATES else "gate_definitions",
                binary_parents=_is_binary_column(samples, a)
                and _is_binary_column(samples, b),
            )
        )
    return duals


def extract_triple_gates(
    edges: pd.DataFrame,
    samples: pd.DataFrame | None = None,
) -> list[TripleGate]:
    triples: list[TripleGate] = []
    for gate, comps in _all_gate_definitions().items():
        if len(comps) != 3:
            continue
        a, b, c = comps
        triples.append(
            TripleGate(
                gate=gate,
                parent_a=a,
                parent_b=b,
                parent_c=c,
                children=_children_of(edges, gate),
                source="old_gates" if gate in OLD_GATES else "gate_definitions",
                binary_parents=all(_is_binary_column(samples, x) for x in comps),
            )
        )
    return triples


def extract_load_nodes() -> list[LoadNode]:
    return [
        LoadNode(load=name, components=tuple(comps))
        for name, comps in LOAD_DEFINITIONS.items()
    ]


def extract_confounded_non_edges(edges: pd.DataFrame) -> list[ConfoundedNonEdge]:
    """X←U→Y with no direct X–Y edge in either direction."""
    succ: dict[str, list[str]] = {}
    edge_set: set[tuple[str, str]] = set()
    for _, row in edges.iterrows():
        s, t = str(row["source"]), str(row["target"])
        succ.setdefault(s, []).append(t)
        edge_set.add((s, t))

    out: list[ConfoundedNonEdge] = []
    for u, children in succ.items():
        uniq = sorted(set(children))
        for x, y in combinations(uniq, 2):
            if (x, y) in edge_set or (y, x) in edge_set:
                continue
            out.append(ConfoundedNonEdge(x=x, y=y, common_cause=u))
    return out


def motif_completion_tasks(dual_gates: Iterable[DualGate]) -> list[MotifCompletionTask]:
    """For each dual gate, two tasks: hide parent_a→gate or parent_b→gate."""
    tasks: list[MotifCompletionTask] = []
    for g in dual_gates:
        if not g.binary_parents:
            continue
        tasks.append(
            MotifCompletionTask(
                gate=g.gate,
                hidden_parent=g.parent_a,
                visible_parent=g.parent_b,
                candidate_source=g.parent_a,
                candidate_target=g.gate,
                children=g.children,
            )
        )
        tasks.append(
            MotifCompletionTask(
                gate=g.gate,
                hidden_parent=g.parent_b,
                visible_parent=g.parent_a,
                candidate_source=g.parent_b,
                candidate_target=g.gate,
                children=g.children,
            )
        )
    return tasks


def latent_gate_triples(dual_gates: Iterable[DualGate]) -> list[tuple[str, str, str, str]]:
    """(parent_a, parent_b, outcome, gate) with gate column masked from HCR."""
    rows: list[tuple[str, str, str, str]] = []
    for g in dual_gates:
        if not g.binary_parents:
            continue
        for child in g.children:
            rows.append((g.parent_a, g.parent_b, child, g.gate))
    return rows


def export_motif_catalog(
    out_dir: Path,
    cfg_or_root: Any = None,
    samples: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Write motif CSVs used by Wave 3B protocols."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes, edges = load_truth_graph(cfg_or_root)

    if samples is None:
        root = nodes_path = _resolve_paths(cfg_or_root)[0].parent
        sample_path = root / "synthetic_pharmacotherapy_v3_samples_clean.csv"
        if sample_path.exists():
            samples = pd.read_csv(sample_path, nrows=5000)

    duals = extract_dual_gates(edges, samples)
    triples = extract_triple_gates(edges, samples)
    loads = extract_load_nodes()
    confounded = extract_confounded_non_edges(edges)
    tasks = motif_completion_tasks(duals)
    latent = latent_gate_triples(duals)

    paths = {
        "dual_gates": out_dir / "wave3b_dual_gates.csv",
        "triple_gates": out_dir / "wave3b_triple_gates.csv",
        "load_nodes": out_dir / "wave3b_load_nodes.csv",
        "confounded_non_edges": out_dir / "wave3b_confounded_non_edges.csv",
        "motif_completion_tasks": out_dir / "wave3b_motif_completion_tasks.csv",
        "latent_gate_triples": out_dir / "wave3b_latent_gate_triples.csv",
    }
    pd.DataFrame([asdict(x) for x in duals]).to_csv(paths["dual_gates"], index=False)
    pd.DataFrame([asdict(x) for x in triples]).to_csv(paths["triple_gates"], index=False)
    pd.DataFrame([asdict(x) for x in loads]).to_csv(paths["load_nodes"], index=False)
    # Confounded set can be large; keep full catalog for hard-negative construction.
    pd.DataFrame([asdict(x) for x in confounded]).to_csv(
        paths["confounded_non_edges"], index=False
    )
    pd.DataFrame([asdict(x) for x in tasks]).to_csv(
        paths["motif_completion_tasks"], index=False
    )
    pd.DataFrame(
        latent,
        columns=["parent_a", "parent_b", "outcome", "masked_gate"],
    ).to_csv(paths["latent_gate_triples"], index=False)

    summary = out_dir / "wave3b_motif_catalog_summary.txt"
    summary.write_text(
        "\n".join(
            [
                "Wave 3B motif catalog",
                f"dual_gates: {len(duals)} (binary_parents={sum(g.binary_parents for g in duals)})",
                f"triple_gates: {len(triples)} (binary_parents={sum(g.binary_parents for g in triples)})",
                f"load_nodes: {len(loads)}",
                f"confounded_non_edges: {len(confounded)}",
                f"motif_completion_tasks (binary dual): {len(tasks)}",
                f"latent_gate_triples: {len(latent)}",
                "",
                "Protocols:",
                "  A motif completion — hide one parent→gate edge; Z=visible co-parent",
                "  B latent gate — HCR3(parent_a, parent_b, outcome) without gate column",
                "  D confounded non-edges — hard negatives for common-cause FPR",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    paths["summary"] = summary
    return paths
