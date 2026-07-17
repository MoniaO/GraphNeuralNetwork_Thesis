#!/usr/bin/env python3
"""Run leakage-aware HCR link-prediction experiments.

This script treats link prediction as the primary task:

    target = edge_label

It never uses audit-only labels such as target_edge_type, target_effect_*,
reference_edge_id or reference_hcr_conditioning_hint as model inputs. Those
columns are merged only after prediction for error analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCORE_MODELS = [
    "hcr_score_only",
    "cmi_score_only",
    "hcr_perm_score",
    "hcr_excess_score",
]

NUMERIC_FEATURES = {
    "structural_only": [
        "source_layer_rank",
        "target_layer_rank",
        "layer_delta",
        "same_layer",
        "target_is_endpoint",
    ],
    "hcr_basic": [
        "conditional_mutual_information",
        "hcr_information_weight",
        "permutation_p_value",
        "hcr_above_null_q95",
        "conditioning_dimension",
        "n_complete",
    ],
    "hcr_extended": [
        "entropy_target",
        "entropy_target_given_source",
        "entropy_target_given_z",
        "mutual_information",
        "conditional_mutual_information",
        "hcr_information_weight",
        "information_removed_by_conditioning",
        "pearson_signed",
        "risk_difference_if_binary_source",
        "permutation_p_value",
        "permutation_null_mean",
        "permutation_null_sd",
        "permutation_null_q95",
        "hcr_above_null_q95",
        "conditioning_dimension",
        "n_complete",
    ],
}
NUMERIC_FEATURES["hcr_plus_structural"] = (
    NUMERIC_FEATURES["hcr_extended"] + NUMERIC_FEATURES["structural_only"]
)

CATEGORICAL_FEATURES = {
    "structural_only": ["source_node_type", "target_node_type"],
    "hcr_basic": [],
    "hcr_extended": [],
    "hcr_plus_structural": ["source_node_type", "target_node_type"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--hcr-dir",
        type=Path,
        default=Path("synthetic_pharmacotherapy_v2_1_nn_benchmark/hcr_link_prediction"),
    )
    p.add_argument("--output-dir", type=Path, default=Path("experiments/link_prediction_outputs"))
    p.add_argument("--full-oracle", action="store_true",
                   help="Use edge_splits.csv / edge_candidates.csv including latent node.")
    p.add_argument("--scenarios", nargs="*", default=None,
                   help="Optional subset, e.g. clean hidden_confounder selection_bias")
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--max-iterations", type=int, default=80)
    return p.parse_args()


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30, 30)
    return 1.0 / (1.0 + np.exp(-x))


def fit_weighted_logistic(
    x: np.ndarray, y: np.ndarray, ridge: float, max_iterations: int
) -> np.ndarray:
    x_design = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(x_design.shape[1], dtype=float)
    prevalence = np.clip(y.mean(), 1e-4, 1 - 1e-4)
    beta[0] = np.log(prevalence / (1 - prevalence))

    positive_weight = max(1.0, float((y == 0).sum()) / max(1, int((y == 1).sum())))
    sample_weight = np.where(y == 1, positive_weight, 1.0)
    penalty = np.eye(x_design.shape[1]) * ridge
    penalty[0, 0] = 0.0

    def objective(candidate: np.ndarray) -> float:
        probability = np.clip(sigmoid(x_design @ candidate), 1e-12, 1 - 1e-12)
        data_loss = -np.sum(
            sample_weight
            * (y * np.log(probability) + (1 - y) * np.log(1 - probability))
        )
        return float(data_loss + 0.5 * candidate @ penalty @ candidate)

    for _ in range(max_iterations):
        probability = sigmoid(x_design @ beta)
        variance_weight = sample_weight * np.maximum(probability * (1 - probability), 1e-6)
        gradient = x_design.T @ (sample_weight * (probability - y)) + penalty @ beta
        hessian = x_design.T @ (variance_weight[:, None] * x_design) + penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient

        old_objective = objective(beta)
        scale = 1.0
        accepted = False
        while scale >= 1e-6:
            proposal = beta - scale * step
            if objective(proposal) <= old_objective + 1e-10:
                beta = proposal
                accepted = True
                break
            scale *= 0.5
        if not accepted or np.max(np.abs(scale * step)) < 1e-7:
            break
    return beta


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    ranked = y[order]
    sorted_score = score[order]
    cumulative = np.cumsum(ranked)
    block_end = np.r_[np.where(sorted_score[:-1] != sorted_score[1:])[0], len(y) - 1]
    true_positives = cumulative[block_end]
    precision = true_positives / (block_end + 1)
    recall = true_positives / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(score).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def classification_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    pred = score >= threshold
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "n": len(y),
        "positives": int(y.sum()),
        "negatives": int((1 - y).sum()),
        "average_precision": average_precision(y, score),
        "roc_auc": roc_auc(y, score),
        "brier_score": float(np.mean((score - y) ** 2)),
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def best_f1_threshold(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return 0.5
    candidates = np.unique(np.quantile(score, np.linspace(0.02, 0.98, 97)))
    best_score, best_threshold = float("-inf"), 0.5
    for threshold in candidates:
        f1 = classification_metrics(y, score, float(threshold))["f1"]
        if f1 > best_score:
            best_score = f1
            best_threshold = float(threshold)
    return best_threshold


def impute_score(values: pd.Series, train_mask: np.ndarray) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    train_values = numeric[train_mask]
    median = np.nanmedian(train_values)
    if not np.isfinite(median):
        median = 0.0
    return np.where(np.isfinite(numeric), numeric, median)


def make_score(group: pd.DataFrame, model: str, train_mask: np.ndarray) -> np.ndarray:
    if model == "hcr_score_only":
        return impute_score(group["hcr_information_weight"], train_mask)
    if model == "cmi_score_only":
        return impute_score(group["conditional_mutual_information"], train_mask)
    if model == "hcr_perm_score":
        hcr = impute_score(group["hcr_information_weight"], train_mask)
        pval = impute_score(group["permutation_p_value"], train_mask)
        return hcr * (1.0 - np.clip(pval, 0.0, 1.0))
    if model == "hcr_excess_score":
        cmi = impute_score(group["conditional_mutual_information"], train_mask)
        q95 = impute_score(group["permutation_null_q95"], train_mask)
        return cmi - q95
    raise ValueError(f"Unknown score model: {model}")


def encode_features(
    frame: pd.DataFrame, model: str, train_mask: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    numeric_columns = NUMERIC_FEATURES[model]
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    medians = np.nanmedian(numeric[train_mask], axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    numeric = np.where(np.isfinite(numeric), numeric, medians)

    means = numeric[train_mask].mean(axis=0)
    scales = numeric[train_mask].std(axis=0)
    scales[scales < 1e-8] = 1.0
    encoded = (numeric - means) / scales
    feature_names = list(numeric_columns)

    cat_columns = []
    for column in CATEGORICAL_FEATURES[model]:
        train_categories = sorted(frame.loc[train_mask, column].astype(str).unique())
        for category in train_categories:
            cat_columns.append((frame[column].astype(str) == category).astype(float).to_numpy())
            feature_names.append(f"{column}={category}")
    if cat_columns:
        encoded = np.column_stack([encoded] + cat_columns)
    return encoded, feature_names


def run_score_model(
    group: pd.DataFrame,
    model: str,
    masks: dict[str, np.ndarray],
    group_key: tuple[str, int, int],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    y = group["edge_label"].to_numpy(dtype=int)
    score = make_score(group, model, masks["train"])
    threshold = best_f1_threshold(y[masks["validation"]], score[masks["validation"]])
    metrics = classification_metrics(y[masks["test"]], score[masks["test"]], threshold)
    scenario, replicate_id, edge_repeat = group_key
    metrics.update(
        {
            "dataset_scenario": scenario,
            "replicate_id": int(replicate_id),
            "edge_repeat": int(edge_repeat),
            "model": model,
        }
    )
    predictions = prediction_rows(group, model, score, threshold, masks["test"])
    return metrics, predictions


def run_logistic_model(
    group: pd.DataFrame,
    model: str,
    masks: dict[str, np.ndarray],
    group_key: tuple[str, int, int],
    ridge: float,
    max_iterations: int,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    y = group["edge_label"].to_numpy(dtype=int)
    x, feature_names = encode_features(group, model, masks["train"])
    beta = fit_weighted_logistic(x[masks["train"]], y[masks["train"]], ridge, max_iterations)
    probability = sigmoid(np.column_stack([np.ones(len(x)), x]) @ beta)
    threshold = best_f1_threshold(y[masks["validation"]], probability[masks["validation"]])
    metrics = classification_metrics(y[masks["test"]], probability[masks["test"]], threshold)

    scenario, replicate_id, edge_repeat = group_key
    metrics.update(
        {
            "dataset_scenario": scenario,
            "replicate_id": int(replicate_id),
            "edge_repeat": int(edge_repeat),
            "model": model,
        }
    )
    predictions = prediction_rows(group, model, probability, threshold, masks["test"])
    coefficients = [
        {
            "dataset_scenario": scenario,
            "replicate_id": int(replicate_id),
            "edge_repeat": int(edge_repeat),
            "model": model,
            "feature": name,
            "coefficient": float(value),
        }
        for name, value in zip(["intercept"] + feature_names, beta)
    ]
    return metrics, predictions, coefficients


def prediction_rows(
    group: pd.DataFrame,
    model: str,
    score: np.ndarray,
    threshold: float,
    test_mask: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for idx in np.where(test_mask)[0]:
        rows.append(
            {
                "dataset_scenario": group.loc[idx, "dataset_scenario"],
                "replicate_id": int(group.loc[idx, "replicate_id"]),
                "edge_repeat": int(group.loc[idx, "edge_repeat"]),
                "candidate_id": group.loc[idx, "candidate_id"],
                "source": group.loc[idx, "source"],
                "target": group.loc[idx, "target"],
                "edge_label": int(group.loc[idx, "edge_label"]),
                "model": model,
                "score": float(score[idx]),
                "threshold": float(threshold),
                "predicted_label": int(score[idx] >= threshold),
            }
        )
    return rows


def add_error_analysis(predictions: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    audit_columns = [
        "candidate_id",
        "target_edge_type",
        "layer_transition",
        "source_node_type",
        "target_node_type",
    ]
    merged = predictions.merge(metadata[audit_columns], on="candidate_id", how="left")
    merged["error_type"] = np.select(
        [
            (merged["edge_label"] == 1) & (merged["predicted_label"] == 1),
            (merged["edge_label"] == 0) & (merged["predicted_label"] == 0),
            (merged["edge_label"] == 0) & (merged["predicted_label"] == 1),
            (merged["edge_label"] == 1) & (merged["predicted_label"] == 0),
        ],
        ["TP", "TN", "FP", "FN"],
        default="unknown",
    )
    return merged


def summarise_errors(enriched_predictions: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    summary = (
        enriched_predictions.groupby(["dataset_scenario", "model"] + by + ["error_type"])
        .size()
        .rename("n")
        .reset_index()
    )
    total = (
        enriched_predictions.groupby(["dataset_scenario", "model"] + by)
        .size()
        .rename("group_total")
        .reset_index()
    )
    summary = summary.merge(total, on=["dataset_scenario", "model"] + by, how="left")
    summary["rate_within_group"] = summary["n"] / summary["group_total"]
    return summary.sort_values(["dataset_scenario", "model"] + by + ["error_type"])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_file = "edge_splits.csv" if args.full_oracle else "edge_splits_observed.csv"
    candidate_file = "edge_candidates.csv" if args.full_oracle else "edge_candidates_observed.csv"
    features = pd.read_csv(args.hcr_dir / "edge_pair_features_train_only.csv")
    splits = pd.read_csv(args.hcr_dir / split_file)
    metadata = pd.read_csv(args.hcr_dir / candidate_file)

    data = features.merge(
        splits[["edge_repeat", "candidate_id", "edge_split"]],
        on="candidate_id",
        how="inner",
    )
    data = data[data["pair_available"] == 1].copy()
    if args.scenarios:
        data = data[data["dataset_scenario"].isin(args.scenarios)].copy()

    metric_rows = []
    prediction_rows_all = []
    coefficient_rows = []
    group_columns = ["dataset_scenario", "replicate_id", "edge_repeat"]

    for group_key, group in data.groupby(group_columns, sort=True):
        group = group.reset_index(drop=True)
        masks = {
            "train": group["edge_split"].eq("train").to_numpy(),
            "validation": group["edge_split"].eq("validation").to_numpy(),
            "test": group["edge_split"].eq("test").to_numpy(),
        }
        y = group["edge_label"].to_numpy(dtype=int)
        if y[masks["train"]].sum() == 0 or (1 - y[masks["train"]]).sum() == 0:
            continue

        for model in SCORE_MODELS:
            metrics, predictions = run_score_model(group, model, masks, group_key)
            metric_rows.append(metrics)
            prediction_rows_all.extend(predictions)

        for model in NUMERIC_FEATURES:
            metrics, predictions, coefficients = run_logistic_model(
                group, model, masks, group_key, args.ridge, args.max_iterations
            )
            metric_rows.append(metrics)
            prediction_rows_all.extend(predictions)
            coefficient_rows.extend(coefficients)

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.DataFrame(prediction_rows_all)
    coefficients_df = pd.DataFrame(coefficient_rows)
    enriched_predictions = add_error_analysis(predictions_df, metadata)

    summary = (
        metrics_df.groupby(["dataset_scenario", "model"])
        .agg(
            n_runs=("average_precision", "size"),
            average_precision_mean=("average_precision", "mean"),
            average_precision_sd=("average_precision", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_sd=("roc_auc", "std"),
            f1_mean=("f1", "mean"),
            f1_sd=("f1", "std"),
            brier_mean=("brier_score", "mean"),
        )
        .reset_index()
    )

    metrics_df.to_csv(args.output_dir / "link_prediction_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "link_prediction_summary.csv", index=False)
    predictions_df.to_csv(args.output_dir / "link_prediction_test_predictions.csv", index=False)
    coefficients_df.to_csv(args.output_dir / "link_prediction_coefficients.csv", index=False)
    enriched_predictions.to_csv(
        args.output_dir / "link_prediction_test_predictions_with_audit_metadata.csv",
        index=False,
    )
    summarise_errors(enriched_predictions, ["target_edge_type"]).to_csv(
        args.output_dir / "error_analysis_by_edge_type.csv",
        index=False,
    )
    summarise_errors(enriched_predictions, ["layer_transition"]).to_csv(
        args.output_dir / "error_analysis_by_layer_transition.csv",
        index=False,
    )

    config = {
        "hcr_dir": str(args.hcr_dir),
        "split_file": split_file,
        "candidate_file": candidate_file,
        "target": "edge_label",
        "scenario_filter": args.scenarios,
        "ridge": args.ridge,
        "max_iterations": args.max_iterations,
        "forbidden_predictor_columns": [
            "edge_label",
            "target_edge_type",
            "target_effect_sign",
            "target_effect_size",
            "reference_edge_id",
            "reference_hcr_conditioning_hint",
        ],
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print(summary.to_string(index=False))
    print(f"\nSaved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
