"""Build the WNERW working DAG G* = G_train ∪ filtered predicted edges."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import networkx as nx
import pandas as pd

# Frozen layer ranks — derived from audited node ``layer`` strings, never G_true.
LAYER_RANK: dict[str, int] = {
    "1_patient_context": 0,
    "patient_context": 0,
    "2_drugs": 1,
    "drug_exposure": 1,
    "3_mechanisms": 2,
    "mechanism": 2,
    "4_intermediate_states": 3,
    "intermediate_state": 3,
    "5_observation_selection": 4,
    "observation": 4,
    "6_endpoints": 5,
    "clinical_endpoint": 5,
}


def layer_rank_from_nodes(nodes: pd.DataFrame) -> dict[str, int]:
    """Map node id → integer layer rank using the audited ``layer`` column."""
    if "node" not in nodes.columns or "layer" not in nodes.columns:
        raise KeyError("nodes frame must contain 'node' and 'layer' columns")
    out: dict[str, int] = {}
    for _, row in nodes.iterrows():
        node = str(row["node"])
        layer = str(row["layer"])
        if layer not in LAYER_RANK:
            raise KeyError(f"Unknown layer {layer!r} for node {node!r}")
        out[node] = LAYER_RANK[layer]
    return out


def _edge_layer_priority(
    source: str,
    target: str,
    layer_rank: Mapping[str, int],
) -> int:
    """Lower = prefer remove first when peeling cycles (reverse < same < forward)."""
    if source not in layer_rank or target not in layer_rank:
        return 0
    rs, rt = layer_rank[source], layer_rank[target]
    if rs > rt:
        return 0
    if rs == rt:
        return 1
    return 2


def _peel_cycles_prefer_nonforward(
    graph: nx.DiGraph,
    layer_rank: Mapping[str, int],
) -> int:
    """Remove cycle edges until DAG; prefer reverse/same-layer edges."""
    removed = 0
    while True:
        try:
            cycle = nx.find_cycle(graph, orientation="original")
        except nx.exception.NetworkXNoCycle:
            break
        best_uv = None
        best_pri = 99
        for u, v, *_ in cycle:
            pri = _edge_layer_priority(str(u), str(v), layer_rank)
            if pri < best_pri:
                best_pri = pri
                best_uv = (u, v)
        if best_uv is None:
            u, v, *_ = cycle[0]
            best_uv = (u, v)
        graph.remove_edge(*best_uv)
        removed += 1
    return removed


def build_wnerw_graph(
    train_edges: pd.DataFrame,
    predicted_edges: pd.DataFrame,
    nodes: pd.DataFrame,
    threshold: float,
    *,
    probability_column: str = "p_calibrated",
    candidate_budget: int | None = None,
    enforce_dag: bool = True,
) -> nx.DiGraph:
    """Construct an acyclic WNERW graph.

    Contract
    --------
    - ``edge_in_train`` / train_edges: **always keep** (self-loops excluded).
      ``topological_allowed`` must NOT gate train edges.
    - Predicted edges: only considered when they pass threshold/budget and
      strict layer order (equivalent to ``topological_allowed``).
    - Cycle cleanup is defensive and prefers removing reverse/same-layer edges.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes["node"].astype(str))

    layer_rank = layer_rank_from_nodes(nodes)

    # --- Train edges: always keep ---
    for source, target in train_edges[["source", "target"]].itertuples(
        index=False, name=None
    ):
        s, t = str(source), str(target)
        if s == t:
            continue
        if s not in graph or t not in graph:
            continue
        graph.add_edge(s, t)

    if enforce_dag and not nx.is_directed_acyclic_graph(graph):
        _peel_cycles_prefer_nonforward(graph, layer_rank)
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Train graph is not a DAG after cycle cleanup.")

    if predicted_edges is None or len(predicted_edges) == 0:
        return graph

    selected = predicted_edges.copy()
    if probability_column not in selected.columns:
        raise KeyError(f"Missing probability column {probability_column!r}")
    selected = selected[selected[probability_column] >= float(threshold)]
    selected = selected.sort_values(probability_column, ascending=False)
    if candidate_budget is not None:
        selected = selected.head(int(candidate_budget))

    # --- Predicted edges: topological_allowed / strict layer order only ---
    for source, target in selected[["source", "target"]].itertuples(
        index=False, name=None
    ):
        s, t = str(source), str(target)
        if s == t:
            continue
        if s not in layer_rank or t not in layer_rank:
            continue
        if layer_rank[s] >= layer_rank[t]:
            continue
        if graph.has_edge(s, t):
            continue
        graph.add_edge(s, t)
        if enforce_dag and not nx.is_directed_acyclic_graph(graph):
            graph.remove_edge(s, t)

    return graph


def train_edges_from_pairs(pairs: Iterable[tuple[str, str]]) -> pd.DataFrame:
    rows = [{"source": s, "target": t} for s, t in pairs]
    return pd.DataFrame(rows, columns=["source", "target"])


def annotate_topological_allowed(
    candidates: pd.DataFrame,
    layer_rank: Mapping[str, int],
) -> pd.Series:
    """Candidate-only mask: source layer strictly precedes target layer.

    This flag is for **selecting new predicted edges**, never for dropping
    ``edge_in_train`` edges from G*.
    """

    def ok(row) -> bool:
        s, t = str(row["source"]), str(row["target"])
        if s not in layer_rank or t not in layer_rank:
            return False
        return layer_rank[s] < layer_rank[t]

    return candidates.apply(ok, axis=1)
