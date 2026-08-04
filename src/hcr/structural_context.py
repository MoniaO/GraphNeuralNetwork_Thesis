"""Structural context selection for Wave 4D (G_train only — never G_true)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import networkx as nx
import pandas as pd

from hcr.motif_registry_v3 import DUAL_GATES, GATE_OUTCOMES, TRIPLE_GATES
from hcr.motifs import GATE_DEFINITIONS, OLD_GATES
from hcr.variable_spec import VariableType
from hcr.variable_specs_v3 import VARIABLE_SPECS

# Never sample gate / interaction nodes as unrestricted context (D2 pool).
_ALL_GATE_NODES = (
    set(DUAL_GATES)
    | set(TRIPLE_GATES)
    | set(GATE_OUTCOMES)
    | set(GATE_DEFINITIONS)
    | set(OLD_GATES)
)

# Layer ranks for temporal precedence (same freeze as WNERW).
LAYER_RANK: dict[str, int] = {
    "1_patient_context": 0,
    "2_drugs": 1,
    "3_mechanisms": 2,
    "4_intermediate_states": 3,
    "5_observation_selection": 4,
    "6_endpoints": 5,
}

ALLOWED_GATE_INPUT_TYPES = {
    "drug_exposure",
    "patient_context",
    "mechanism",
    "adr_or_intermediate_state",
}


@dataclass(frozen=True)
class StructuralSelectorAudit:
    """Anti-leakage record — selector must never touch G_true / test labels."""

    graph_source: str = "train"
    has_access_to_true_graph: bool = False
    uses_test_labels: bool = False


def g_train_digraph_from_hetero(data) -> nx.DiGraph:
    """Recover directed G_train name graph from HeteroData (skip reverse edges)."""
    g = nx.DiGraph()
    for ntype in data.node_types:
        names = getattr(data[ntype], "node_name", None)
        if names is None:
            continue
        g.add_nodes_from(str(n) for n in names)
    for etype in data.edge_types:
        src_t, rel, dst_t = etype
        if str(rel).startswith("rev_"):
            continue
        edge_index = data[etype].edge_index
        if edge_index is None or edge_index.numel() == 0:
            continue
        src_names = [str(x) for x in data[src_t].node_name]
        dst_names = [str(x) for x in data[dst_t].node_name]
        for i in range(edge_index.size(1)):
            g.add_edge(src_names[int(edge_index[0, i])], dst_names[int(edge_index[1, i])])
    return g


def load_node_metadata(cfg: Any) -> dict[str, dict[str, Any]]:
    from hcr.motifs import load_truth_graph

    nodes, _ = load_truth_graph(cfg)
    out: dict[str, dict[str, Any]] = {}
    for _, row in nodes.iterrows():
        node = str(row["node"])
        layer = str(row["layer"])
        out[node] = {
            "node_type": str(row.get("node_type", "")),
            "layer": layer,
            "layer_rank": LAYER_RANK.get(layer, -1),
        }
    return out


def temporally_precedes(meta_z: Mapping[str, Any], meta_gate: Mapping[str, Any]) -> bool:
    """Allow same-layer parents (common for mechanism↔mechanism gate components)."""
    rz = int(meta_z.get("layer_rank", -1))
    rg = int(meta_gate.get("layer_rank", -1))
    if rz < 0 or rg < 0:
        return False
    return rz <= rg


def allowed_gate_input_type(node_type: str) -> bool:
    return str(node_type) in ALLOWED_GATE_INPUT_TYPES


def select_structural_contexts(
    graph_train: nx.DiGraph,
    source: str,
    target_gate: str,
    downstream: str | None,
    node_metadata: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Z ∈ Pa_{G_train}(G) \\ {X} with temporal / descendant / mediator filters."""
    if target_gate not in graph_train:
        return []
    contexts: list[str] = []
    for z in graph_train.predecessors(target_gate):
        z = str(z)
        if z == source:
            continue
        meta_z = node_metadata.get(z)
        meta_g = node_metadata.get(target_gate)
        if meta_z is None or meta_g is None:
            continue
        if not allowed_gate_input_type(str(meta_z["node_type"])):
            continue
        if nx.has_path(graph_train, target_gate, z):
            continue
        if downstream is not None and downstream in graph_train:
            if nx.has_path(graph_train, downstream, z):
                continue
        # Direct mediator source→Z→gate. (Full has_path would also kill true
        # co-parents linked via load/burden nodes, e.g. ckd→polypharmacy→nsaid.)
        if source in graph_train and graph_train.has_edge(source, z):
            continue
        if not temporally_precedes(meta_z, meta_g):
            continue
        contexts.append(z)
    return sorted(contexts)


def select_structural_context_map(
    cfg: Any,
    train_data,
    tasks,
) -> dict[tuple[str, str], str | None]:
    """Map held-out (X,G) → chosen structural Z (first sorted candidate) or None."""
    g_train = g_train_digraph_from_hetero(train_data)
    meta = load_node_metadata(cfg)
    audit = StructuralSelectorAudit()
    assert audit.graph_source == "train"
    assert not audit.has_access_to_true_graph
    assert not audit.uses_test_labels

    out: dict[tuple[str, str], str | None] = {}
    for t in tasks:
        y = GATE_OUTCOMES.get(t.gate)
        if y is None and t.children:
            y = t.children[0]
        cands = select_structural_contexts(
            g_train,
            source=t.candidate_source,
            target_gate=t.gate,
            downstream=y,
            node_metadata=meta,
        )
        out[(t.candidate_source, t.candidate_target)] = cands[0] if cands else None
    return out


def all_binary_context_candidates(
    source: str,
    gate: str,
    downstream: str | None,
) -> list[str]:
    """Unsafe D2 pool: all binary nodes except source/gate/Y/other gates."""
    forbidden = {source, gate} | _ALL_GATE_NODES
    if downstream:
        forbidden.add(downstream)
    return sorted(
        n
        for n, spec in VARIABLE_SPECS.items()
        if spec.variable_type == VariableType.BINARY and n not in forbidden
    )


def score_context_on_train(
    train_patients: pd.DataFrame,
    source: str,
    context: str,
    downstream: str,
) -> float:
    """Simple association score for D2 top-1 (train patients only)."""
    from hcr.binary_features import binary_pair_features

    if (
        source not in train_patients.columns
        or context not in train_patients.columns
        or downstream not in train_patients.columns
    ):
        return float("-inf")
    try:
        xz = binary_pair_features(
            train_patients[source].to_numpy(),
            train_patients[context].to_numpy(),
        )
        zy = binary_pair_features(
            train_patients[context].to_numpy(),
            train_patients[downstream].to_numpy(),
        )
        xy = binary_pair_features(
            train_patients[source].to_numpy(),
            train_patients[downstream].to_numpy(),
        )
    except ValueError:
        return float("-inf")
    return float(abs(xz.phi) + abs(zy.phi) + 0.25 * abs(xy.phi))


def select_all_context_top1_map(
    cfg: Any,
    train_patients: pd.DataFrame,
    tasks,
) -> dict[tuple[str, str], str | None]:
    out: dict[tuple[str, str], str | None] = {}
    for t in tasks:
        y = GATE_OUTCOMES.get(t.gate) or (t.children[0] if t.children else None)
        if y is None:
            out[(t.candidate_source, t.candidate_target)] = None
            continue
        best_z, best_s = None, float("-inf")
        for z in all_binary_context_candidates(t.candidate_source, t.gate, y):
            s = score_context_on_train(train_patients, t.candidate_source, z, y)
            if s > best_s:
                best_s, best_z = s, z
        out[(t.candidate_source, t.candidate_target)] = best_z
    return out


def matched_random_context(
    source: str,
    gate: str,
    downstream: str | None,
    structural_z: str | None,
    node_metadata: Mapping[str, Mapping[str, Any]],
    seed: int,
) -> str | None:
    """Match layer of structural Z when possible; else any allowed binary."""
    import hashlib

    forbidden = {source, gate} | _ALL_GATE_NODES
    if downstream:
        forbidden.add(downstream)
    if structural_z:
        forbidden.add(structural_z)
    target_layer = None
    if structural_z and structural_z in node_metadata:
        target_layer = node_metadata[structural_z].get("layer")
    pool = [
        n
        for n, spec in VARIABLE_SPECS.items()
        if spec.variable_type == VariableType.BINARY
        and n not in forbidden
        and (
            target_layer is None
            or node_metadata.get(n, {}).get("layer") == target_layer
        )
    ]
    if not pool:
        pool = [
            n
            for n, spec in VARIABLE_SPECS.items()
            if spec.variable_type == VariableType.BINARY and n not in forbidden
        ]
    if not pool:
        return None
    digest = hashlib.sha256(f"d4|{source}|{gate}|{seed}".encode()).hexdigest()
    return pool[int(digest[:8], 16) % len(pool)]
