"""I/O helpers for scored-edge dumps used by the coparent registry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_edges(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    # One row per directed edge for graph construction / scoring.
    return df.drop_duplicates(subset=["source", "target"], keep="first").reset_index(
        drop=True
    )


def merge_patient_shuffle(
    edges: pd.DataFrame,
    patient_shuffle_path: Path,
    *,
    source_col: str = "p_hcr2_calibrated",
    out_col: str = "p_patient_shuffle_calibrated",
) -> pd.DataFrame:
    ps = load_edges(patient_shuffle_path)
    out = edges.copy()
    merged = out.merge(
        ps[["candidate_id", source_col]].rename(columns={source_col: out_col}),
        on="candidate_id",
        how="left",
    )
    # Fallback if candidate_id missing/mismatch: join on endpoints.
    if merged[out_col].isna().all():
        merged = out.merge(
            ps[["source", "target", source_col]].rename(columns={source_col: out_col}),
            on=["source", "target"],
            how="left",
        )
    return merged


def load_queries(path: Path) -> pd.DataFrame:
    q = pd.read_csv(path)
    if "query_split" not in q.columns:
        # Frozen 50/50 valid/test split by query_id (deterministic).
        rng = np.random.default_rng(20260722)
        order = np.arange(len(q))
        rng.shuffle(order)
        split = np.array(["valid"] * len(q), dtype=object)
        n_test = len(q) // 2
        split[order[:n_test]] = "test"
        q = q.copy()
        q["query_split"] = split
    return q


def resolve_path(pattern: str, seed: int, root: Path) -> Path:
    path = Path(pattern.format(seed=seed))
    if not path.is_absolute():
        path = root / path
    return path
