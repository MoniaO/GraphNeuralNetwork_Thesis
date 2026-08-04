"""Subgroup AUPRC audits for Wave 7 Panel B (validation selection slices)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from hcr.variable_spec import VariableType
from hcr.variable_specs_v3 import VARIABLE_SPECS


def _kind(name: str) -> str:
    spec = VARIABLE_SPECS.get(str(name))
    if spec is None:
        return "unknown"
    if spec.variable_type == VariableType.BINARY:
        return "binary"
    if spec.variable_type == VariableType.COUNT:
        return "count"
    return "continuous"


def pair_bucket(u: str, v: str) -> dict[str, bool]:
    ku, kv = _kind(u), _kind(v)
    kinds = {ku, kv}
    return {
        "binary_binary": ku == "binary" and kv == "binary",
        "count_involved": "count" in kinds,
        "continuous_involved": "continuous" in kinds,
        "mixed_type": ku != kv and "unknown" not in kinds,
    }


def safe_auprc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if len(y) == 0 or np.unique(y).size < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def summarize_prediction_frame(
    frame: pd.DataFrame,
    *,
    label_col: str = "label",
    prob_col: str = "probability",
    source_col: str = "source",
    target_col: str = "target",
    supported_col: str | None = "hcr_supported",
    motif_col: str | None = "hcr_motif_available",
    hard_neg_col: str | None = "is_hard_negative",
    held_out_col: str | None = "is_held_out_positive",
) -> dict[str, float]:
    """Compute overall + typed subgroup metrics from a candidate prediction frame."""
    y = frame[label_col].to_numpy(dtype=float)
    p = frame[prob_col].to_numpy(dtype=float)
    out: dict[str, float] = {
        "n_all": float(len(frame)),
        "auprc_all": safe_auprc(y, p),
        "brier_all": float(brier_score_loss(y, p)) if len(y) else float("nan"),
    }

    masks = {
        "binary_binary": [],
        "count_involved": [],
        "continuous_involved": [],
        "mixed_type": [],
    }
    for u, v in zip(frame[source_col].astype(str), frame[target_col].astype(str)):
        b = pair_bucket(u, v)
        for k in masks:
            masks[k].append(b[k])
    for k, flags in masks.items():
        m = np.asarray(flags, dtype=bool)
        out[f"auprc_{k}"] = safe_auprc(y[m], p[m]) if m.any() else float("nan")
        out[f"n_{k}"] = float(m.sum())

    if supported_col and supported_col in frame.columns:
        m = frame[supported_col].astype(bool).to_numpy()
        out["auprc_fully_supported"] = safe_auprc(y[m], p[m]) if m.any() else float("nan")
        out["n_fully_supported"] = float(m.sum())
    if motif_col and motif_col in frame.columns:
        # fallback single-pair = motif unavailable
        m = ~frame[motif_col].astype(bool).to_numpy()
        out["auprc_fallback_single_pair"] = safe_auprc(y[m], p[m]) if m.any() else float("nan")
        out["n_fallback_single_pair"] = float(m.sum())
    if hard_neg_col and hard_neg_col in frame.columns:
        m = frame[hard_neg_col].astype(bool).to_numpy()
        # evaluate hard negatives vs all positives
        pos = y >= 0.5
        use = pos | m
        out["auprc_hard_negatives"] = safe_auprc(y[use], p[use]) if use.any() else float("nan")
    if held_out_col and held_out_col in frame.columns:
        m = frame[held_out_col].astype(bool).to_numpy()
        out["auprc_held_out_positives"] = safe_auprc(y[m], p[m]) if m.any() else float("nan")
        out["n_held_out_positives"] = float(m.sum())
    return out
