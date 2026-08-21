"""Patient-conditioned edge energies for Wave 5C."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import networkx as nx
import numpy as np
import pandas as pd

from .patient_activity import gate_activation

EPSILON = 1e-6


@dataclass(frozen=True)
class PatientEnergyConfig:
    temperature: float = 0.5
    length_penalty: float = 0.1
    node_activity_weight: float = 0.25
    gate_weight: float = 1.0
    static_hcr_weight: float = 0.0
    hcr_probability_column: str = "p_hcr2_calibrated"
    # Optional remapping of context columns (C5 context shuffle).
    context_column_map: tuple[tuple[str, str], ...] = ()


def _remap_activity(
    patient_activity: Mapping[str, float],
    context_column_map: tuple[tuple[str, str], ...],
) -> dict[str, float]:
    activity = {str(k): float(v) for k, v in patient_activity.items()}
    if not context_column_map:
        return activity
    # Map logical context node → values from substitute column key.
    for logical, substitute in context_column_map:
        activity[str(logical)] = float(
            patient_activity.get(str(substitute), activity.get(str(logical), 0.0))
        )
    return activity


def build_patient_edge_scores(
    graph: nx.DiGraph,
    edge_evidence: pd.DataFrame,
    patient_activity: Mapping[str, float],
    admissible_activity_nodes: set[str],
    context_map: Mapping[tuple[str, str], tuple[str, ...]],
    config: PatientEnergyConfig,
) -> dict[tuple[str, str], float]:
    if config.temperature <= 0:
        raise ValueError("Temperature must be positive.")

    activity = _remap_activity(patient_activity, config.context_column_map)
    indexed = edge_evidence.drop_duplicates(
        subset=["source", "target"], keep="first"
    ).set_index(["source", "target"])

    scores: dict[tuple[str, str], float] = {}
    for source, target in graph.edges:
        source, target = str(source), str(target)
        edge = (source, target)
        if edge not in indexed.index:
            # Train-only residual: length penalty only.
            scores[edge] = float((-config.length_penalty) / config.temperature)
            continue
        row = indexed.loc[edge]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        node_activity = 0.0
        if target in admissible_activity_nodes:
            node_activity = float(activity.get(target, 0.0))

        motif_activation = 0.0
        if edge in context_map:
            motif_activation = gate_activation(
                source=source,
                context_nodes=context_map[edge],
                patient_activity=activity,
            )

        static_prior = 0.0
        if config.static_hcr_weight > 0 and not bool(row.get("edge_in_train", False)):
            col = config.hcr_probability_column
            if col in row.index and pd.notna(row[col]):
                probability = float(np.clip(float(row[col]), EPSILON, 1.0 - EPSILON))
                static_prior = config.static_hcr_weight * float(np.log(probability))

        raw_score = (
            -config.length_penalty
            + config.node_activity_weight * node_activity
            + config.gate_weight * motif_activation
            + static_prior
        )
        scores[edge] = float(raw_score / config.temperature)
    return scores
