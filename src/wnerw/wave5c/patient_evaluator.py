"""Evaluate one patient × query under Wave 5C energies."""

from __future__ import annotations

from typing import Mapping

import networkx as nx
import pandas as pd

from wnerw.finite_path_ensemble import FinitePathEnsemble
from wnerw.topk_paths import top_k_paths as dp_top_k_paths
from wnerw.wave5b.exact_metrics import (
    backward_log_partition,
    count_paths,
    exact_hub_mass,
    exact_path_entropy,
    true_path_mass,
)
from wnerw.wave5c.edge_marginals import exact_edge_mass
from wnerw.wave5c.patient_activity import gate_activation
from wnerw.wave5c.patient_energy import PatientEnergyConfig, build_patient_edge_scores


def evaluate_patient_query(
    patient_id: str,
    patient_activity: Mapping[str, float],
    query: Mapping[str, object],
    graph: nx.DiGraph,
    edge_evidence: pd.DataFrame,
    context_map: dict[tuple[str, str], tuple[str, ...]],
    admissible_nodes: set[str],
    hubs: set[str],
    config: PatientEnergyConfig,
    *,
    top_k: int = 100,
    skip_topk: bool = False,
) -> dict[str, object]:
    source = str(query["source"])
    endpoint = str(query["endpoint"])
    hidden_edge = (
        str(query["hidden_edge_source"]),
        str(query["hidden_edge_target"]),
    )
    true_path = [
        str(n).strip()
        for n in str(query.get("true_path", "")).split(">")
        if str(n).strip()
    ]
    contexts = context_map.get(hidden_edge, ())
    if not contexts and "context_nodes" in query:
        from .patient_queries import parse_context_nodes

        contexts = parse_context_nodes(query["context_nodes"])

    source_active = float(patient_activity.get(source, 0.0))
    gate_active = gate_activation(source, contexts, patient_activity)
    context_active_count = int(
        sum(1 for c in contexts if float(patient_activity.get(c, 0.0)) >= 0.5)
    )

    base = {
        "patient_id": patient_id,
        "query_id": str(query["query_id"]),
        "source": source,
        "endpoint": endpoint,
        "gate": str(query.get("gate", "")),
        "source_active": source_active,
        "gate_active": gate_active,
        "context_active_count": context_active_count,
        "has_path": False,
        "hidden_edge_mass": 0.0,
        "true_path_mass": 0.0,
        "true_path_rank_at_100": float("inf"),
        "reciprocal_rank_at_100": 0.0,
        "hidden_edge_hit_at_1": 0,
        "hidden_edge_hit_at_5": 0,
        "path_entropy": 0.0,
        "effective_path_count": 0,
        "hub_mass": 0.0,
    }

    scores = build_patient_edge_scores(
        graph=graph,
        edge_evidence=edge_evidence,
        patient_activity=patient_activity,
        admissible_activity_nodes=admissible_nodes,
        context_map=context_map,
        config=config,
    )

    if source not in graph or endpoint not in graph:
        return base
    if not nx.has_path(graph, source, endpoint):
        return base

    backward = backward_log_partition(graph, scores, endpoint)
    hidden_mass = exact_edge_mass(
        hidden_edge=hidden_edge,
        graph=graph,
        edge_scores=scores,
        backward=backward,
        source=source,
    )
    route_mass = true_path_mass(true_path, graph, scores, backward)
    n_paths = count_paths(graph, source, endpoint)
    rank = float("inf")
    rr = 0.0
    he1 = he5 = 0
    if not skip_topk:
        # DP top-K (supports positive q_uv from gate bonuses).
        ens = FinitePathEnsemble(graph, scores, temperature=1.0)
        ranked = dp_top_k_paths(
            ens,
            source,
            endpoint,
            k=top_k,
            log_partition=ens.backward_log_partition(endpoint),
        )
        path_nodes = [list(p.nodes) for p in ranked]
        try:
            rank = float(path_nodes.index(true_path) + 1)
        except ValueError:
            rank = float("inf")
        import math

        rr = 0.0 if not math.isfinite(rank) else 1.0 / rank

        def he_hit(k: int) -> int:
            return int(
                any(
                    hidden_edge in list(zip(p[:-1], p[1:]))
                    for p in path_nodes[:k]
                )
            )

        he1, he5 = he_hit(1), he_hit(5)

    base.update(
        {
            "has_path": True,
            "hidden_edge_mass": float(hidden_mass),
            "true_path_mass": float(route_mass),
            "true_path_rank_at_100": rank,
            "reciprocal_rank_at_100": rr,
            "hidden_edge_hit_at_1": he1,
            "hidden_edge_hit_at_5": he5,
            "path_entropy": exact_path_entropy(
                graph, scores, backward, source, endpoint
            ),
            "effective_path_count": int(n_paths),
            "hub_mass": exact_hub_mass(
                graph, scores, backward, source, endpoint, hubs
            ),
        }
    )
    return base
