"""Summaries for Task A E3 per-edge-type results.

The raw per-seed table often marks almost every relation as
``insufficient_support`` when the per-seed threshold is n_positive >= 10.
For this split that threshold is too strict (max ≈ 8 positives / seed).

We therefore emit two reports from the same raw CSV:

1. Exploratory — relations with mean per-seed n_positive >= 4
2. Main — mean±SD across seeds for relations with pooled
   support (sum of n_positive over seeds) >= ``main_pooled_min``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Relations with enough per-seed mass to discuss carefully on this split.
PRIORITY_EDGE_TYPES = (
    "drug_to_burden",
    "risk_modifier",
    "interaction_amplify",
    "mechanism_to_adr",
    "adr_burden_component",
    "observation_process",
    "drug_to_load",
    "drug_to_mechanism",
)


def _seed_level_pivot(
    frame: pd.DataFrame,
    *,
    split: str = "valid",
) -> pd.DataFrame:
    """One row per (model, seed, edge_type) with profile AUPRCs and deltas."""
    subset = frame[frame["split"].astype(str) == split].copy()
    if subset.empty:
        return subset

    support = (
        subset[subset["profile"] == "empirical"][
            ["model", "training_seed", "edge_type", "n", "n_positive", "prevalence"]
        ]
        .drop_duplicates(subset=["model", "training_seed", "edge_type"])
        .rename(
            columns={
                "n": "n_emp",
                "n_positive": "n_positive_emp",
                "prevalence": "prevalence_emp",
            }
        )
    )

    pivot = subset.pivot_table(
        index=["model", "training_seed", "edge_type"],
        columns="profile",
        values="auprc",
        aggfunc="first",
    ).reset_index()

    if {"empirical", "topology_only"}.issubset(pivot.columns):
        pivot["delta_emp_top"] = pivot["empirical"] - pivot["topology_only"]
    if {"empirical", "empirical_shuffled"}.issubset(pivot.columns):
        pivot["delta_emp_shuf"] = pivot["empirical"] - pivot["empirical_shuffled"]

    return pivot.merge(
        support,
        on=["model", "training_seed", "edge_type"],
        how="left",
    )


def _aggregate_by_model_edge(seed_level: pd.DataFrame) -> pd.DataFrame:
    if seed_level.empty:
        return seed_level

    named = {
        "n_seeds": ("training_seed", "nunique"),
        "mean_n_positive": ("n_positive_emp", "mean"),
        "sum_n_positive": ("n_positive_emp", "sum"),
        "mean_prevalence": ("prevalence_emp", "mean"),
    }
    optional_mean_std = [
        ("empirical", "mean_emp", "std_emp"),
        ("topology_only", "mean_topology", "std_topology"),
        ("empirical_shuffled", "mean_shuffled", "std_shuffled"),
        ("delta_emp_top", "mean_delta_emp_top", "sd_delta_emp_top"),
        ("delta_emp_shuf", "mean_delta_emp_shuf", "sd_delta_emp_shuf"),
    ]
    for src, mean_name, std_name in optional_mean_std:
        if src in seed_level.columns:
            named[mean_name] = (src, "mean")
            named[std_name] = (src, "std")

    aggregated = (
        seed_level.groupby(["model", "edge_type"], dropna=False)
        .agg(**named)
        .reset_index()
    )

    for src, count_name in (
        ("delta_emp_top", "n_pos_delta_emp_top"),
        ("delta_emp_shuf", "n_pos_delta_emp_shuf"),
    ):
        if src in seed_level.columns:
            counts = (
                seed_level.groupby(["model", "edge_type"])[src]
                .apply(lambda s: int((s > 0).sum()))
                .reset_index(name=count_name)
            )
            aggregated = aggregated.merge(counts, on=["model", "edge_type"], how="left")

    aggregated["auprc_baseline"] = aggregated["mean_prevalence"]
    if "mean_emp" in aggregated.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            lift = aggregated["mean_emp"] / aggregated["mean_prevalence"]
        aggregated["mean_auprc_lift"] = lift.where(
            aggregated["mean_prevalence"].gt(0)
            & np.isfinite(lift.to_numpy(dtype=float))
        )
    return aggregated


def summarize_per_edge_type(
    frame: pd.DataFrame,
    *,
    split: str = "valid",
    exploratory_min_positives: float = 4.0,
    main_pooled_min_positives: float = 20.0,
    priority_edge_types: tuple[str, ...] = PRIORITY_EDGE_TYPES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (seed_level, exploratory_summary, main_summary)."""
    seed_level = _seed_level_pivot(frame, split=split)
    aggregated = _aggregate_by_model_edge(seed_level)
    if aggregated.empty:
        return seed_level, aggregated.copy(), aggregated.copy()

    aggregated["priority_relation"] = aggregated["edge_type"].isin(priority_edge_types)
    aggregated["exploratory"] = (
        aggregated["mean_n_positive"] >= float(exploratory_min_positives)
    )
    aggregated["main_support"] = (
        aggregated["sum_n_positive"] >= float(main_pooled_min_positives)
    )

    exploratory = aggregated[aggregated["exploratory"]].copy()
    exploratory["report_tier"] = "exploratory"
    exploratory["interpretation_note"] = (
        "Per-seed n_positive usually < 10; treat as exploratory."
    )

    main = aggregated[aggregated["main_support"]].copy()
    main["report_tier"] = "main"
    main["interpretation_note"] = (
        "Mean±SD across seeds; pooled n_positive meets main threshold. "
        "Not a single pooled-prediction table (raw logits not stored)."
    )

    sort_cols = ["priority_relation"]
    ascending = [False]
    if "mean_delta_emp_top" in aggregated.columns:
        sort_cols.append("mean_delta_emp_top")
        ascending.append(False)
    else:
        sort_cols.append("edge_type")
        ascending.append(True)

    exploratory = exploratory.sort_values(sort_cols, ascending=ascending)
    main = main.sort_values(sort_cols, ascending=ascending)
    return seed_level, exploratory, main


def write_per_edge_type_summaries(
    raw_csv: Path,
    *,
    split: str = "valid",
) -> dict[str, Path]:
    frame = pd.read_csv(raw_csv)
    seed_level, exploratory, main = summarize_per_edge_type(frame, split=split)

    stem = raw_csv.with_suffix("")
    paths = {
        "seed_level": Path(f"{stem}_seed_level.csv"),
        "exploratory": Path(f"{stem}_summary_exploratory.csv"),
        "main": Path(f"{stem}_summary.csv"),
    }
    seed_level.to_csv(paths["seed_level"], index=False)
    exploratory.to_csv(paths["exploratory"], index=False)
    main.to_csv(paths["main"], index=False)
    return paths
