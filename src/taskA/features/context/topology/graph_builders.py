"""Graph selection used when building the coparent-Z registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import networkx as nx
import pandas as pd

from taskA.features.context.graph_builder import _peel_cycles_prefer_nonforward, layer_rank_from_nodes


@dataclass(frozen=True)
class SelectionSpec:
    name: str
    probability_column: str
    threshold: float


def selected_edge_mask(
    edges: pd.DataFrame,
    probability_column: str,
    threshold: float,
) -> pd.Series:
    """Train edges always kept; predicted only if topo-allowed and p ≥ threshold."""
    edge_in_train = edges["edge_in_train"].astype(bool)
    predicted_allowed = (
        ~edge_in_train
        & edges["topological_allowed"].astype(bool)
        & edges[probability_column].notna()
        & (edges[probability_column] >= float(threshold))
    )
    return edge_in_train | predicted_allowed


def _add_selected_edges(graph: nx.DiGraph, selected: pd.DataFrame) -> None:
    for row in selected.itertuples(index=False):
        s, t = str(row.source), str(row.target)
        if s == t:
            continue
        graph.add_edge(
            s,
            t,
            edge_in_train=bool(row.edge_in_train),
            edge_type=str(getattr(row, "edge_type", "unknown")),
        )


def _ensure_dag(
    graph: nx.DiGraph,
    *,
    layer_rank: Mapping[str, int] | None = None,
) -> nx.DiGraph:
    if nx.is_directed_acyclic_graph(graph):
        return graph
    if layer_rank is None:
        # Fallback: peel arbitrary cycle edges until DAG.
        while True:
            try:
                cycle = nx.find_cycle(graph, orientation="original")
            except nx.exception.NetworkXNoCycle:
                break
            u, v, *_ = cycle[0]
            graph.remove_edge(u, v)
    else:
        _peel_cycles_prefer_nonforward(graph, layer_rank)
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Selected model graph is not a DAG after cleanup.")
    return graph


def build_model_graph(
    edges: pd.DataFrame,
    probability_column: str,
    threshold: float,
    *,
    nodes: pd.DataFrame | None = None,
) -> nx.DiGraph:
    """Own-model topology graph (Panel T)."""
    mask = selected_edge_mask(
        edges=edges,
        probability_column=probability_column,
        threshold=threshold,
    )
    graph = nx.DiGraph()
    if nodes is not None and "node" in nodes.columns:
        graph.add_nodes_from(nodes["node"].astype(str))
    selected = edges.loc[
        mask, ["source", "target", "edge_in_train", "edge_type"]
    ].drop_duplicates(subset=["source", "target"], keep="first")
    _add_selected_edges(graph, selected)
    layer_rank = layer_rank_from_nodes(nodes) if nodes is not None else None
    return _ensure_dag(graph, layer_rank=layer_rank)


def build_train_only_graph(
    edges: pd.DataFrame,
    *,
    nodes: pd.DataFrame | None = None,
) -> nx.DiGraph:
    """T0 — message-passing train edges only."""
    graph = nx.DiGraph()
    if nodes is not None and "node" in nodes.columns:
        graph.add_nodes_from(nodes["node"].astype(str))
    selected = edges.loc[
        edges["edge_in_train"].astype(bool),
        ["source", "target", "edge_in_train", "edge_type"],
    ].drop_duplicates(subset=["source", "target"], keep="first")
    _add_selected_edges(graph, selected)
    layer_rank = layer_rank_from_nodes(nodes) if nodes is not None else None
    return _ensure_dag(graph, layer_rank=layer_rank)


def build_fixed_union_graph(
    edges: pd.DataFrame,
    specs: list[SelectionSpec],
    *,
    nodes: pd.DataFrame | None = None,
) -> nx.DiGraph:
    """Panel E shared graph: train ∪ any predicted edge selected by ≥1 model."""
    keep = edges["edge_in_train"].astype(bool).copy()
    for spec in specs:
        keep |= selected_edge_mask(
            edges=edges,
            probability_column=spec.probability_column,
            threshold=spec.threshold,
        )
    graph = nx.DiGraph()
    if nodes is not None and "node" in nodes.columns:
        graph.add_nodes_from(nodes["node"].astype(str))
    selected = edges.loc[
        keep, ["source", "target", "edge_in_train", "edge_type"]
    ].drop_duplicates(subset=["source", "target"], keep="first")
    _add_selected_edges(graph, selected)
    layer_rank = layer_rank_from_nodes(nodes) if nodes is not None else None
    return _ensure_dag(graph, layer_rank=layer_rank)


def hub_nodes_from_degree(
    graph: nx.DiGraph,
    *,
    degree_quantile: float = 0.90,
) -> set[str]:
    if graph.number_of_nodes() == 0:
        return set()
    deg = {str(n): int(graph.degree(n)) for n in graph.nodes}
    if not deg:
        return set()
    thr = float(pd.Series(deg).quantile(degree_quantile))
    return {n for n, d in deg.items() if d >= thr}
