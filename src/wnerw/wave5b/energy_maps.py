"""Probability → additive path energy maps for Wave 5B Panel E."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

EPSILON = 1e-6


@dataclass(frozen=True)
class EnergyConfig:
    probability_column: str
    temperature: float = 1.0
    length_penalty: float = 0.0
    uncertainty_penalty: float = 0.0
    uncertainty_column: str = "hcr_uncertainty"


def clipped_probability(value: float) -> float:
    return float(np.clip(value, EPSILON, 1.0 - EPSILON))


def uniform_edge_scores(graph: nx.DiGraph) -> dict[tuple[str, str], float]:
    """Panel T / E0: q_uv = 0 for every edge (equal path weights)."""
    return {(str(u), str(v)): 0.0 for u, v in graph.edges}


def length_only_edge_scores(
    graph: nx.DiGraph,
    *,
    length_penalty: float,
    temperature: float = 1.0,
) -> dict[tuple[str, str], float]:
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")
    raw = -float(length_penalty)
    return {(str(u), str(v)): raw / float(temperature) for u, v in graph.edges}


def build_edge_energies(
    graph: nx.DiGraph,
    edges: pd.DataFrame,
    config: EnergyConfig,
) -> dict[tuple[str, str], float]:
    """Return additive q_uv already divided by temperature.

    Train edges: log-evidence = 0 (still pay length penalty).
    Predicted: log(p) − λ_U · uncertainty − λ_L, then / T.
    """
    if config.temperature <= 0:
        raise ValueError("Temperature must be positive.")

    indexed = (
        edges.drop_duplicates(subset=["source", "target"], keep="first")
        .set_index(["source", "target"])
    )
    scores: dict[tuple[str, str], float] = {}

    for source, target in graph.edges:
        key = (str(source), str(target))
        if key not in indexed.index:
            raise KeyError(f"Missing edge evidence for {key}")
        row = indexed.loc[key]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        if bool(row["edge_in_train"]):
            log_evidence = 0.0
            uncertainty_cost = 0.0
        else:
            probability = clipped_probability(float(row[config.probability_column]))
            log_evidence = float(np.log(probability))
            uncertainty = float(
                np.clip(row.get(config.uncertainty_column, 0.0) or 0.0, 0.0, 1.0)
            )
            uncertainty_cost = config.uncertainty_penalty * uncertainty

        raw_score = log_evidence - uncertainty_cost - config.length_penalty
        scores[key] = raw_score / config.temperature

    return scores


def add_matched_edgewise_shuffle(
    edges: pd.DataFrame,
    source_column: str,
    output_column: str,
    seed: int,
) -> pd.DataFrame:
    """Matched edgewise shuffle for Panel E (topology fixed, scores permuted)."""
    result = edges.copy()
    rng = np.random.default_rng(int(seed))
    predicted = (
        ~result["edge_in_train"].astype(bool)
        & result["topological_allowed"].astype(bool)
    )
    result[output_column] = result[source_column]

    uncertainty = result.loc[predicted, "hcr_uncertainty"].astype(float)
    try:
        uncertainty_bin = pd.qcut(
            uncertainty, q=5, labels=False, duplicates="drop"
        )
    except ValueError:
        uncertainty_bin = pd.Series(0, index=uncertainty.index)

    result.loc[predicted, "_uncertainty_bin"] = uncertainty_bin
    selected = result.loc[predicted].copy()
    group_columns = ["edge_type", "_uncertainty_bin"]
    for _, group in selected.groupby(group_columns, dropna=False):
        indices = group.index.to_numpy()
        if len(indices) < 2:
            continue
        shuffled_values = rng.permutation(
            result.loc[indices, source_column].to_numpy()
        )
        result.loc[indices, output_column] = shuffled_values

    return result.drop(columns=["_uncertainty_bin"], errors="ignore")
