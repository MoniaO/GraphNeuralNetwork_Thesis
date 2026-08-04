"""Role-specific and path-recovery metrics for Wave 5D."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def safe_auprc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    if np.unique(y).size < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def safe_auroc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    if np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def edge_classification_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
) -> dict[str, float]:
    y = np.asarray(labels).astype(int)
    p = np.clip(np.asarray(probs).astype(float), 1e-6, 1 - 1e-6)
    out = {
        "auprc": safe_auprc(y, p),
        "auroc": safe_auroc(y, p),
        "brier": float(brier_score_loss(y, p)) if len(y) else float("nan"),
        "log_loss": float(log_loss(y, p)) if np.unique(y).size > 1 else float("nan"),
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "prevalence": float(y.mean()) if len(y) else float("nan"),
    }
    return out


def role_specific_fpr(
    labels: np.ndarray,
    probs: np.ndarray,
    roles: list[str],
    threshold: float = 0.5,
) -> pd.DataFrame:
    y = np.asarray(labels).astype(int)
    p = np.asarray(probs).astype(float)
    pred = (p >= threshold).astype(int)
    rows = []
    for role in sorted(set(roles)):
        mask = np.array([r == role for r in roles])
        # FPR only among true negatives of this role
        neg = mask & (y == 0)
        if neg.sum() == 0:
            fpr = float("nan")
        else:
            fpr = float(pred[neg].mean())
        rows.append(
            {
                "negative_role": role,
                "n_neg": int(neg.sum()),
                "fpr": fpr,
                "auprc_role": safe_auprc(y[mask], p[mask]) if mask.sum() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def hits_at_k(
    labels: np.ndarray,
    scores: np.ndarray,
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, float]:
    """Global ranking Hits@K / MRR over positives among all candidates."""
    y = np.asarray(labels).astype(int)
    s = np.asarray(scores).astype(float)
    order = np.argsort(-s)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(s) + 1)
    pos_ranks = ranks[y == 1]
    out: dict[str, float] = {}
    if len(pos_ranks) == 0:
        out["mrr"] = float("nan")
        for k in ks:
            out[f"hits_at_{k}"] = float("nan")
        return out
    out["mrr"] = float(np.mean(1.0 / pos_ranks))
    for k in ks:
        out[f"hits_at_{k}"] = float(np.mean(pos_ranks <= k))
    return out
