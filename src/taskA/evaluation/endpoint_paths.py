"""Downstream reachability: node G ⇝ clinical endpoint (Task A audit).

Reachability is computed on the audited DAG **without** using any candidate
edge A→G. Only paths that start at G (or G itself if G is an endpoint).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd

DEFAULT_ENDPOINTS = (
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
)


def load_audited_digraph(
    edges_csv: Path,
    *,
    nodes_csv: Path | None = None,
) -> nx.DiGraph:
    edges = pd.read_csv(edges_csv)
    g = nx.DiGraph()
    if nodes_csv is not None and Path(nodes_csv).exists():
        nodes = pd.read_csv(nodes_csv)
        if "is_latent" in nodes.columns:
            latent = nodes["is_latent"].astype(str).str.lower().isin({"true", "1"})
            nodes = nodes.loc[~latent]
        g.add_nodes_from(nodes["node"].astype(str).tolist())
    for _, row in edges.iterrows():
        g.add_edge(str(row["source"]), str(row["target"]))
    return g


def reachability_record(
    graph: nx.DiGraph,
    start: str,
    endpoint: str,
    *,
    path_cutoff: int = 12,
) -> dict:
    start = str(start)
    endpoint = str(endpoint)
    if start not in graph or endpoint not in graph:
        return {
            "is_downstream_reachable": False,
            "shortest_path_length": np.nan,
            "number_of_downstream_paths": 0,
        }
    if start == endpoint:
        return {
            "is_downstream_reachable": True,
            "shortest_path_length": 0,
            "number_of_downstream_paths": 1,
        }
    if not nx.has_path(graph, start, endpoint):
        return {
            "is_downstream_reachable": False,
            "shortest_path_length": np.nan,
            "number_of_downstream_paths": 0,
        }
    length = int(nx.shortest_path_length(graph, start, endpoint))
    # Cap enumeration for large DAGs; count is informative, not exact for huge fans.
    n_paths = sum(1 for _ in nx.all_simple_paths(graph, start, endpoint, cutoff=path_cutoff))
    return {
        "is_downstream_reachable": True,
        "shortest_path_length": length,
        "number_of_downstream_paths": int(n_paths),
    }


def build_node_endpoint_table(
    graph: nx.DiGraph,
    endpoints: Iterable[str] = DEFAULT_ENDPOINTS,
    *,
    path_cutoff: int = 12,
) -> pd.DataFrame:
    endpoints = [str(e) for e in endpoints]
    rows = []
    for node in sorted(graph.nodes()):
        for ep in endpoints:
            rec = reachability_record(graph, node, ep, path_cutoff=path_cutoff)
            rows.append({"target_node": node, "endpoint": ep, **rec})
    return pd.DataFrame(rows)


def expand_candidates_to_endpoint_registry(
    candidates: pd.DataFrame,
    node_endpoint: pd.DataFrame,
    *,
    source_col: str = "source",
    target_col: str = "target",
    id_col: str | None = "candidate_id",
) -> pd.DataFrame:
    """Join candidates to node→endpoint reachability on target_node."""
    need = {source_col, target_col}
    missing = need - set(candidates.columns)
    if missing:
        raise KeyError(f"candidates missing {missing}")
    left = candidates.copy()
    left[source_col] = left[source_col].astype(str)
    left[target_col] = left[target_col].astype(str)
    if id_col is None or id_col not in left.columns:
        left["candidate_id"] = [
            f"{s}__{t}" for s, t in zip(left[source_col], left[target_col])
        ]
        id_col = "candidate_id"
    merged = left.merge(
        node_endpoint,
        left_on=target_col,
        right_on="target_node",
        how="left",
    )
    merged["source_node"] = merged[source_col]
    cols = [
        id_col,
        "source_node",
        "target_node",
        "endpoint",
        "is_downstream_reachable",
        "shortest_path_length",
        "number_of_downstream_paths",
    ]
    if "edge_label" in merged.columns:
        cols.append("edge_label")
    if "split" in merged.columns:
        cols.append("split")
    return merged[cols].rename(columns={id_col: "candidate_edge_id"})


def safe_auprc(labels: np.ndarray, probs: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    labels = np.asarray(labels).astype(float)
    probs = np.asarray(probs).astype(float)
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(average_precision_score(labels, probs))


def safe_auroc(labels: np.ndarray, probs: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels).astype(float)
    probs = np.asarray(probs).astype(float)
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, probs))


def safe_brier(labels: np.ndarray, probs: np.ndarray) -> float:
    from sklearn.metrics import brier_score_loss

    labels = np.asarray(labels).astype(float)
    probs = np.asarray(probs).astype(float)
    if labels.size == 0:
        return float("nan")
    return float(brier_score_loss(labels, probs))


def endpoint_subset_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    shortest_lengths: np.ndarray | None = None,
) -> dict:
    labels = np.asarray(labels).astype(float)
    probs = np.asarray(probs).astype(float)
    n = int(labels.size)
    n_pos = int(labels.sum()) if n else 0
    pi = float(labels.mean()) if n else float("nan")
    auprc = safe_auprc(labels, probs)
    lift = float(auprc / pi) if pi and pi > 0 and np.isfinite(auprc) else float("nan")
    nauprc = (
        float((auprc - pi) / (1.0 - pi))
        if np.isfinite(auprc) and np.isfinite(pi) and pi < 1.0
        else float("nan")
    )
    mean_len = float("nan")
    if shortest_lengths is not None and n:
        sl = np.asarray(shortest_lengths, dtype=float)
        sl = sl[np.isfinite(sl)]
        if sl.size:
            mean_len = float(sl.mean())
    return {
        "n_candidate_edges": n,
        "n_positive_edges": n_pos,
        "edge_prevalence": pi,
        "auprc": auprc,
        "auroc": safe_auroc(labels, probs),
        "brier": safe_brier(labels, probs),
        "auprc_lift": lift,
        "normalized_auprc": nauprc,
        "mean_shortest_path_length": mean_len,
    }
