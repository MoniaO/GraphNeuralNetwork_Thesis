"""Causal-role registry for Wave 4D Panel B (G_true evaluator-only)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import networkx as nx
import numpy as np
import pandas as pd

from hcr.motifs import load_truth_graph
from hcr.variable_spec import VariableType
from hcr.variable_specs_v3 import VARIABLE_SPECS


@dataclass(frozen=True)
class ContextRoleRecord:
    source: str
    target: str
    context: str
    downstream: str | None
    role: str
    edge_label: int


def truth_digraph(cfg: Any = None) -> nx.DiGraph:
    nodes, edges = load_truth_graph(cfg)
    g = nx.DiGraph()
    g.add_nodes_from(nodes["node"].astype(str).tolist())
    for _, row in edges.iterrows():
        g.add_edge(str(row["source"]), str(row["target"]))
    return g


def build_true_coparent_records(graph_true: nx.DiGraph) -> list[ContextRoleRecord]:
    records: list[ContextRoleRecord] = []
    for target in graph_true.nodes:
        parents = list(graph_true.predecessors(target))
        for source in parents:
            for context in parents:
                if source == context:
                    continue
                records.append(
                    ContextRoleRecord(
                        source=str(source),
                        target=str(target),
                        context=str(context),
                        downstream=None,
                        role="true_coparent",
                        edge_label=1,
                    )
                )
    return records


def build_confounder_records(graph_true: nx.DiGraph) -> list[ContextRoleRecord]:
    records: list[ContextRoleRecord] = []
    for context in graph_true.nodes:
        children = list(graph_true.successors(context))
        for source in children:
            for target in children:
                if source == target:
                    continue
                if graph_true.has_edge(source, target) or graph_true.has_edge(target, source):
                    continue
                records.append(
                    ContextRoleRecord(
                        source=str(source),
                        target=str(target),
                        context=str(context),
                        downstream=None,
                        role="confounder",
                        edge_label=0,
                    )
                )
    return records


def build_mediator_records(graph_true: nx.DiGraph) -> list[ContextRoleRecord]:
    """X→Z→Y without direct X→Y."""
    records: list[ContextRoleRecord] = []
    for z in graph_true.nodes:
        for source in graph_true.predecessors(z):
            for target in graph_true.successors(z):
                if source == target:
                    continue
                if graph_true.has_edge(source, target):
                    continue
                records.append(
                    ContextRoleRecord(
                        source=str(source),
                        target=str(target),
                        context=str(z),
                        downstream=None,
                        role="mediator",
                        edge_label=0,
                    )
                )
    return records


def build_collider_records(graph_true: nx.DiGraph) -> list[ContextRoleRecord]:
    """X→Z←Y without X–Y edge."""
    records: list[ContextRoleRecord] = []
    for z in graph_true.nodes:
        parents = list(graph_true.predecessors(z))
        for i, source in enumerate(parents):
            for target in parents[i + 1 :]:
                if graph_true.has_edge(source, target) or graph_true.has_edge(target, source):
                    continue
                records.append(
                    ContextRoleRecord(
                        source=str(source),
                        target=str(target),
                        context=str(z),
                        downstream=None,
                        role="collider",
                        edge_label=0,
                    )
                )
                records.append(
                    ContextRoleRecord(
                        source=str(target),
                        target=str(source),
                        context=str(z),
                        downstream=None,
                        role="collider",
                        edge_label=0,
                    )
                )
    return records


def build_descendant_records(graph_true: nx.DiGraph) -> list[ContextRoleRecord]:
    """Post-outcome context: Y→Z; score candidate is a parent→Y edge if any, else skip.

    Records (source=parent of Y, target=Y, context=Z) when Y→Z exists and
    source→Y exists — Z is temporally after Y and must not be selected.
    """
    records: list[ContextRoleRecord] = []
    for y in graph_true.nodes:
        descendants = list(graph_true.successors(y))
        if not descendants:
            continue
        parents = list(graph_true.predecessors(y))
        for source in parents:
            for z in descendants:
                if source == z:
                    continue
                records.append(
                    ContextRoleRecord(
                        source=str(source),
                        target=str(y),
                        context=str(z),
                        downstream=str(y),
                        role="descendant",
                        edge_label=0,
                    )
                )
    return records


def _node_type(name: str) -> str:
    spec = VARIABLE_SPECS.get(name)
    if spec is None:
        return "unknown"
    return str(spec.variable_type.value if hasattr(spec.variable_type, "value") else spec.variable_type)


def _prevalence(train_patients: pd.DataFrame | None, name: str) -> float:
    if train_patients is None or name not in train_patients.columns:
        return 0.5
    col = train_patients[name].astype(float)
    return float(col.mean()) if len(col) else 0.5


def _support_bucket(train_patients: pd.DataFrame | None, a: str, b: str) -> str:
    if train_patients is None or a not in train_patients.columns or b not in train_patients.columns:
        return "unknown"
    both = ((train_patients[a].astype(float) > 0.5) & (train_patients[b].astype(float) > 0.5)).sum()
    if both >= 30:
        return "high"
    if both >= 5:
        return "mid"
    return "low"


def _match_key(
    rec: ContextRoleRecord,
    train_patients: pd.DataFrame | None,
) -> tuple:
    return (
        _node_type(rec.source),
        _node_type(rec.target),
        round(_prevalence(train_patients, rec.source), 1),
        round(_prevalence(train_patients, rec.target), 1),
        _support_bucket(train_patients, rec.source, rec.target),
    )


def subsample_role(
    records: list[ContextRoleRecord],
    max_per_role: int,
    train_patients: pd.DataFrame | None,
    seed: int,
    reference_keys: set[tuple] | None = None,
) -> list[ContextRoleRecord]:
    """Cap role size; prefer records matching reference keys when provided."""
    if not records:
        return []
    rng = np.random.default_rng(seed)
    n = min(max_per_role, len(records))
    if reference_keys:
        matched = [r for r in records if _match_key(r, train_patients) in reference_keys]
        rest = [r for r in records if r not in matched]
        take = []
        if matched:
            idx = rng.choice(len(matched), size=min(n, len(matched)), replace=False)
            take.extend(matched[i] for i in idx)
        need = n - len(take)
        if need > 0 and rest:
            idx = rng.choice(len(rest), size=min(need, len(rest)), replace=False)
            take.extend(rest[i] for i in idx)
        return take
    idx = rng.choice(len(records), size=n, replace=False)
    return [records[i] for i in idx]


def build_context_role_registry(
    cfg: Any = None,
    *,
    train_patients: pd.DataFrame | None = None,
    max_per_role: int = 50,
    seed: int = 20260722,
    roles: Iterable[str] | None = None,
) -> list[ContextRoleRecord]:
    """Build balanced role registry from G_true (evaluator only)."""
    g = truth_digraph(cfg)
    wanted = set(roles or (
        "true_coparent",
        "confounder",
        "mediator",
        "collider",
        "descendant",
    ))
    builders = {
        "true_coparent": build_true_coparent_records,
        "confounder": build_confounder_records,
        "mediator": build_mediator_records,
        "collider": build_collider_records,
        "descendant": build_descendant_records,
    }
    raw: dict[str, list[ContextRoleRecord]] = {}
    for role, fn in builders.items():
        if role in wanted:
            raw[role] = fn(g)

    # Prefer type/prevalence match to true_coparent keys.
    ref_keys = {_match_key(r, train_patients) for r in raw.get("true_coparent", [])}
    out: list[ContextRoleRecord] = []
    for role, recs in raw.items():
        # Prefer binary endpoints when possible.
        binaryish = [
            r
            for r in recs
            if VARIABLE_SPECS.get(r.source, None) is not None
            and VARIABLE_SPECS.get(r.target, None) is not None
            and VARIABLE_SPECS[r.source].variable_type == VariableType.BINARY
            and VARIABLE_SPECS[r.target].variable_type == VariableType.BINARY
        ]
        pool = binaryish if len(binaryish) >= 10 else recs
        sampled = subsample_role(
            pool,
            max_per_role=max_per_role,
            train_patients=train_patients,
            seed=seed + hash(role) % 10_000,
            reference_keys=ref_keys if role != "true_coparent" else None,
        )
        out.extend(sampled)
    return out


def registry_to_frame(records: list[ContextRoleRecord]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in records])


def classify_selected_z_role(
    graph_true: nx.DiGraph,
    source: str,
    target: str,
    z: str | None,
    downstream: str | None,
) -> str:
    """Label what role the selected context plays w.r.t. (source, target[, Y])."""
    if z is None:
        return "none"
    if z not in graph_true:
        return "unknown"
    parents = set(graph_true.predecessors(target))
    if z in parents and source in parents:
        return "true_coparent"
    if (
        graph_true.has_edge(z, source)
        and graph_true.has_edge(z, target)
        and not graph_true.has_edge(source, target)
    ):
        return "confounder"
    if graph_true.has_edge(source, z) and graph_true.has_edge(z, target):
        return "mediator"
    if graph_true.has_edge(source, z) and graph_true.has_edge(target, z):
        return "collider"
    if downstream and graph_true.has_edge(downstream, z):
        return "descendant"
    if graph_true.has_edge(target, z):
        return "descendant"
    return "other"
