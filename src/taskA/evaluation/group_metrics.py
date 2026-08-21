"""Group-wise Task A metrics (per edge_type / node-type pair)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)


def safe_auroc(labels, probabilities):
    if np.unique(labels).size < 2:
        return np.nan
    return float(roc_auc_score(labels, probabilities))


def metrics_for_group(
    labels,
    probabilities,
    threshold: float,
    min_positives: int = 10,
):
    labels = np.asarray(labels).astype(float)
    probabilities = np.asarray(probabilities).astype(float)
    predictions = (probabilities >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )

    beta = 2.0
    if precision + recall == 0:
        f2 = 0.0
    else:
        f2 = (
            (1 + beta**2)
            * precision
            * recall
            / (beta**2 * precision + recall)
        )

    n_positive = int(labels.sum())
    prevalence = float(labels.mean()) if len(labels) else float("nan")
    auprc = (
        float(average_precision_score(labels, probabilities))
        if np.unique(labels).size > 1
        else float("nan")
    )
    auprc_lift = (
        float(auprc / prevalence) if prevalence and prevalence > 0 and np.isfinite(auprc) else float("nan")
    )

    return {
        "n": int(len(labels)),
        "n_positive": n_positive,
        "prevalence": prevalence,
        "auprc": auprc,
        "auprc_baseline": prevalence,
        "auprc_lift": auprc_lift,
        "auroc": safe_auroc(labels, probabilities),
        "brier": float(brier_score_loss(labels, probabilities)) if len(labels) else float("nan"),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f2": float(f2),
        "insufficient_support": bool(n_positive < min_positives),
    }


def grouped_metrics(
    prediction_frame: pd.DataFrame,
    group_column: str,
    threshold: float,
    min_positives: int = 10,
):
    rows = []

    for group_value, group in prediction_frame.groupby(group_column):
        metrics = metrics_for_group(
            labels=group["label"].to_numpy(),
            probabilities=group["probability"].to_numpy(),
            threshold=threshold,
            min_positives=min_positives,
        )
        rows.append({group_column: group_value, **metrics})

    return pd.DataFrame(rows)


def relation_metrics_with_shared_negatives(
    prediction_frame: pd.DataFrame,
    threshold: float,
    edge_type_column: str = "edge_type",
    negative_marker: str = "negative",
    min_positives: int = 10,
):
    """Per-relation metrics using a shared negative pool.

    Candidate negatives are not typed by audited relation. For each
    audited ``edge_type`` *r* we therefore evaluate:

        positives with edge_type == r  ∪  all negatives

    so AUPRC / AUROC remain well-defined.
    """
    required = {"label", "probability", edge_type_column}
    missing = required - set(prediction_frame.columns)
    if missing:
        raise KeyError(f"prediction_frame missing columns: {sorted(missing)}")

    negatives = prediction_frame[
        prediction_frame[edge_type_column].astype(str) == negative_marker
    ]
    relations = sorted(
        {
            value
            for value in prediction_frame[edge_type_column].astype(str).unique()
            if value != negative_marker
        }
    )

    rows = []
    for relation in relations:
        positives = prediction_frame[
            prediction_frame[edge_type_column].astype(str) == relation
        ]
        subset = pd.concat([positives, negatives], ignore_index=True)
        metrics = metrics_for_group(
            labels=subset["label"].to_numpy(),
            probabilities=subset["probability"].to_numpy(),
            threshold=threshold,
            min_positives=min_positives,
        )
        rows.append(
            {
                edge_type_column: relation,
                "n_negatives_shared": int(len(negatives)),
                **metrics,
            }
        )

    return pd.DataFrame(rows)
