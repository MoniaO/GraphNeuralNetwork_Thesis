"""Structural edge-context registry (G_train parents only — never G_true)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import networkx as nx
import pandas as pd

from taskA.features.pair_basis.motif_registry_v3 import DUAL_GATES, GATE_OUTCOMES, TRIPLE_GATES
from taskA.features.pair_basis.structural_context import (
    g_train_digraph_from_hetero,
    load_node_metadata,
    select_structural_contexts,
)


@dataclass(frozen=True)
class EdgeContext:
    source: str
    target: str
    context_nodes: tuple[str, ...]


ALLOWED_CONTEXT_TYPES = {"drug_exposure", "patient_context"}
GATE_LIKE_TYPES = {"mechanism", "adr_or_intermediate_state"}


def build_edge_context_registry(
    graph_train: nx.DiGraph,
    predicted_edges: pd.DataFrame,
    node_metadata: dict[str, dict[str, Any]],
    *,
    seed: int,
    working_graph: nx.DiGraph | None = None,
) -> pd.DataFrame:
    """Contexts C_AG from G_train co-parents; selector_source never true_graph.

    Includes parent→gate edges present in the working G* (train or predicted),
    so gate bonuses apply to the mechanistic A→G edges used in coparent queries.
    """
    records: list[dict[str, object]] = []
    node_type = {n: str(m.get("node_type", "")) for n, m in node_metadata.items()}

    edge_meta = predicted_edges.drop_duplicates(
        subset=["source", "target"], keep="first"
    ).set_index(["source", "target"])

    # Candidate gate edges: known dual/triple parents present in G* (or G_train).
    gate_parent_edges: list[tuple[str, str, bool]] = []
    all_gates = {**DUAL_GATES, **TRIPLE_GATES}
    host = working_graph if working_graph is not None else graph_train
    for gate, parents in all_gates.items():
        if gate not in host:
            continue
        for parent in parents:
            parent = str(parent)
            if parent in host and host.has_edge(parent, gate):
                in_train = False
                if (parent, gate) in edge_meta.index:
                    in_train = bool(edge_meta.loc[(parent, gate)]["edge_in_train"])
                gate_parent_edges.append((parent, str(gate), in_train))

    # Also keep predicted topo-allowed mechanism edges (broader pool).
    pred = predicted_edges.loc[
        (~predicted_edges["edge_in_train"].astype(bool))
        & predicted_edges["topological_allowed"].astype(bool)
    ].drop_duplicates(subset=["source", "target"], keep="first")
    seen = {(s, t) for s, t, _ in gate_parent_edges}
    for row in pred.itertuples(index=False):
        source, target = str(row.source), str(row.target)
        if (source, target) in seen:
            continue
        if node_type.get(target) not in GATE_LIKE_TYPES:
            continue
        gate_parent_edges.append((source, target, False))

    for source, target, in_train in gate_parent_edges:
        if node_type.get(target) not in GATE_LIKE_TYPES and target not in all_gates:
            continue

        downstream = GATE_OUTCOMES.get(target)
        # Known dual/triple co-parents are the canonical AND contexts.
        # They may include load/risk parents beyond drug_exposure/patient_context;
        # those values are used ONLY for gate activation, not as free node bonuses.
        if target in DUAL_GATES:
            contexts = [p for p in DUAL_GATES[target] if p != source]
            selector_source = "expert_gate_schema"
        elif target in TRIPLE_GATES:
            contexts = [p for p in TRIPLE_GATES[target] if p != source]
            selector_source = "expert_gate_schema"
        else:
            contexts = select_structural_contexts(
                graph_train,
                source=source,
                target_gate=target,
                downstream=downstream,
                node_metadata=node_metadata,
            )
            contexts = [
                c for c in contexts if node_type.get(c) in ALLOWED_CONTEXT_TYPES
            ]
            selector_source = "graph_train_structural_contexts"
            if not contexts and target in graph_train:
                contexts = sorted(
                    str(p)
                    for p in graph_train.predecessors(target)
                    if str(p) != source
                    and node_type.get(str(p)) in ALLOWED_CONTEXT_TYPES
                )
                selector_source = "graph_train_parents"
        # Never use endpoints / the gate itself as context.
        contexts = [
            str(c)
            for c in contexts
            if str(c) not in {source, target}
            and node_type.get(str(c))
            not in {"clinical_endpoint", "observation_or_selection"}
        ]
        if not contexts:
            continue

        # Frozen order for ctxs[0] selection downstream.
        contexts = sorted(str(c) for c in contexts)
        selected_context = contexts[0]

        etype = "unknown"
        if (source, target) in edge_meta.index:
            etype = str(edge_meta.loc[(source, target)].get("edge_type", "unknown"))

        records.append(
            {
                "seed": int(seed),
                "source": source,
                "target": target,
                "edge_type": etype,
                "edge_in_train": bool(in_train),
                "is_predicted_gate_edge": (not in_train),
                "context_nodes": json.dumps(contexts),
                "context_candidates": json.dumps(contexts),
                "context_count": len(contexts),
                "context_rank": 0,
                "context_selection_rule": "lexicographic_first_of_sorted_candidates",
                "selected_context": selected_context,
                "selector_source": selector_source,
            }
        )

    df = pd.DataFrame.from_records(records)
    if len(df) and "true_graph" in set(df["selector_source"].astype(str).unique()):
        raise AssertionError("Registry must not use true_graph selector")
    return df

def build_context_map(
    registry: pd.DataFrame,
) -> dict[tuple[str, str], tuple[str, ...]]:
    context_map: dict[tuple[str, str], tuple[str, ...]] = {}
    if registry is None or len(registry) == 0:
        return context_map
    for row in registry.itertuples(index=False):
        nodes = tuple(str(n) for n in json.loads(row.context_nodes))
        context_map[(str(row.source), str(row.target))] = nodes
    return context_map


def admissible_patient_feature_table(
    node_metadata: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for node, meta in sorted(node_metadata.items()):
        ntype = str(meta.get("node_type", ""))
        allowed = ntype in ALLOWED_CONTEXT_TYPES
        rows.append(
            {
                "node": node,
                "node_type": ntype,
                "layer": str(meta.get("layer", "")),
                "admissible": allowed,
            }
        )
    return pd.DataFrame(rows)


def g_train_from_evidence_and_hetero(
    train_data,
) -> nx.DiGraph:
    return g_train_digraph_from_hetero(train_data)
