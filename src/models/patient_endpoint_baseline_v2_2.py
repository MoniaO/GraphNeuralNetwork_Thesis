#!/usr/bin/env python3
"""Leakage-aware patient-level endpoint baseline for v2.2.

Task B only: predict clinical endpoints from patient rows.
This is separate from structural link prediction (Task A / edge_label).

Protocol:
  * train one binary model per endpoint
  * run each scenario separately (never concatenate scenarios)
  * reuse the frozen patient split from ``2. Data/splits/patient_splits_v2_2.csv``
  * fit on train, select threshold-free metrics on validation, report test once
  * default features are exposure/context only (no endpoints, no observation proxies)

Loss / model:
  * logistic regression with balanced class weights
  * equivalent objective to weighted binary cross-entropy / log-loss
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCENARIOS = [
    "clean",
    "hidden_confounder",
    "selection_bias",
    "no_overlap",
    "noisy_documentation",
    "multihospital",
]

DEFAULT_ENDPOINTS = [
    "AKI",
    "Falls",
    "Depression",
    "DILI",
    "Hospitalization",
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "2. Data" / "dataset_v2_2",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=PROJECT_ROOT / "2. Data" / "splits" / "patient_splits_v2_2.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "5. Results and reports"
        / "patient_endpoint_v2_2_outputs",
    )
    parser.add_argument("--scenarios", nargs="+", default=["clean"])
    parser.add_argument("--endpoints", nargs="+", default=DEFAULT_ENDPOINTS)
    parser.add_argument(
        "--feature-set",
        choices=["exposure_context", "with_mechanisms"],
        default="exposure_context",
        help=(
            "exposure_context: patient context + drugs + interaction gates + "
            "active_drug_count. with_mechanisms also adds mechanism nodes, "
            "but still excludes endpoints and observation/selection proxies."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--max-iter", type=int, default=2000)
    return parser.parse_args()


def _boolean(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def load_nodes(data_dir: Path) -> pd.DataFrame:
    nodes = pd.read_csv(data_dir / "synthetic_pharmacotherapy_v2_2_nodes.csv")
    return nodes.loc[~_boolean(nodes["is_latent"])].copy()


def feature_columns(
    nodes: pd.DataFrame,
    sample_columns: list[str],
    feature_set: str,
) -> list[str]:
    allowed_types = {
        "patient_context",
        "drug_exposure",
        "mechanism",
    }
    if feature_set == "exposure_context":
        # Keep only interaction gates / counts from mechanism layer.
        mechanism_keep = {
            "active_drug_count",
            "ddi_renal_double_hit",
            "ddi_renal_triple_whammy",
            "ddi_cns_depression_synergy",
            "ddi_bleeding_dual",
            "ddi_bleeding_triple",
            "ddi_serotonergic_synergy",
            "ddi_qt_multidrug_load",
            "ddi_hepatic_triple_hit",
        }
    else:
        mechanism_keep = set(
            nodes.loc[nodes["node_type"].eq("mechanism"), "node"].astype(str)
        )

    selected: list[str] = []
    for record in nodes.to_dict(orient="records"):
        name = str(record["node"])
        node_type = str(record["node_type"])
        if node_type not in allowed_types:
            continue
        if node_type == "mechanism" and name not in mechanism_keep:
            continue
        if name not in sample_columns:
            continue
        selected.append(name)

    # Preserve a stable, inspectable order.
    return sorted(selected)


def resolve_feature_matrix(
    frame: pd.DataFrame,
    feature_names: list[str],
    scenario: str,
) -> tuple[pd.DataFrame, list[str]]:
    columns: list[str] = []
    used: list[str] = []
    for name in feature_names:
        recorded = f"recorded_{name}"
        if scenario == "noisy_documentation" and recorded in frame.columns:
            columns.append(recorded)
            used.append(recorded)
        else:
            columns.append(name)
            used.append(name)
    matrix = frame[columns].apply(pd.to_numeric, errors="coerce")
    return matrix, used


def metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    if len(np.unique(y_true)) < 2:
        return {
            "ap": float("nan"),
            "roc_auc": float("nan"),
            "brier": float("nan"),
            "prevalence": float(y_true.mean()) if len(y_true) else float("nan"),
            "n": int(len(y_true)),
            "n_positive": int(y_true.sum()),
        }
    return {
        "ap": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "prevalence": float(y_true.mean()),
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
    }


def fit_endpoint(
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    seed: int,
    max_iter: int,
) -> Pipeline:
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=max_iter,
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )
    model.fit(train_x, train_y)
    return model


def run_one(
    data_dir: Path,
    split_file: Path,
    scenario: str,
    endpoint: str,
    feature_set: str,
    seed: int,
    max_iter: int,
) -> tuple[pd.DataFrame, dict[str, object], list[str], pd.DataFrame, pd.DataFrame]:
    nodes = load_nodes(data_dir)
    samples = pd.read_csv(
        data_dir / f"synthetic_pharmacotherapy_v2_2_samples_{scenario}.csv"
    )
    splits = pd.read_csv(split_file, dtype={"patient_id": str})
    samples["patient_id"] = samples["patient_id"].astype(str)
    frame = samples.merge(splits[["patient_id", "split"]], on="patient_id", how="inner")
    if frame["split"].isna().any():
        raise RuntimeError(f"{scenario}: missing split labels after merge")
    if endpoint not in frame.columns:
        raise KeyError(f"{endpoint} missing from {scenario}")

    feature_names = feature_columns(nodes, frame.columns.tolist(), feature_set)
    if not feature_names:
        raise RuntimeError("No usable feature columns selected")

    summary_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    used_features: list[str] | None = None
    model: Pipeline | None = None

    for split_name in ["train", "validation", "test"]:
        part = frame.loc[frame["split"].eq(split_name)].copy()
        x_part, used = resolve_feature_matrix(part, feature_names, scenario)
        y_part = pd.to_numeric(part[endpoint], errors="coerce").fillna(0).astype(int)
        if used_features is None:
            used_features = used
        if split_name == "train":
            model = fit_endpoint(x_part, y_part.to_numpy(), seed, max_iter)
        assert model is not None
        scores = model.predict_proba(x_part)[:, 1]
        split_metrics = metrics(y_part.to_numpy(), scores)
        summary_rows.append(
            {
                "scenario": scenario,
                "endpoint": endpoint,
                "feature_set": feature_set,
                "split": split_name,
                **split_metrics,
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "patient_id": part["patient_id"].to_numpy(),
                    "scenario": scenario,
                    "endpoint": endpoint,
                    "feature_set": feature_set,
                    "split": split_name,
                    "y_true": y_part.to_numpy(),
                    "y_score": scores,
                }
            )
        )

    assert used_features is not None and model is not None
    coef = model.named_steps["clf"].coef_.ravel()
    coef_frame = pd.DataFrame(
        {
            "scenario": scenario,
            "endpoint": endpoint,
            "feature_set": feature_set,
            "feature": used_features,
            "coefficient": coef,
        }
    ).sort_values("coefficient", key=np.abs, ascending=False)

    train_metrics = next(row for row in summary_rows if row["split"] == "train")
    val_metrics = next(row for row in summary_rows if row["split"] == "validation")
    test_metrics = next(row for row in summary_rows if row["split"] == "test")
    headline = {
        "scenario": scenario,
        "endpoint": endpoint,
        "feature_set": feature_set,
        "model": "logistic_regression_balanced",
        "loss": "log_loss_with_balanced_class_weights",
        "n_features": len(used_features),
        "train_ap": train_metrics["ap"],
        "validation_ap": val_metrics["ap"],
        "test_ap": test_metrics["ap"],
        "train_roc_auc": train_metrics["roc_auc"],
        "validation_roc_auc": val_metrics["roc_auc"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_brier": test_metrics["brier"],
        "test_prevalence": test_metrics["prevalence"],
        "test_n": test_metrics["n"],
        "test_n_positive": test_metrics["n_positive"],
        "random_ap_baseline": test_metrics["prevalence"],
    }
    return (
        pd.concat(prediction_frames, ignore_index=True),
        headline,
        used_features,
        coef_frame,
        pd.DataFrame(summary_rows),
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    headlines: list[dict[str, object]] = []
    all_predictions: list[pd.DataFrame] = []
    all_split_metrics: list[pd.DataFrame] = []
    all_coefficients: list[pd.DataFrame] = []
    feature_catalog: dict[str, list[str]] = {}

    for scenario in args.scenarios:
        for endpoint in args.endpoints:
            (
                predictions,
                headline,
                used_features,
                coef_frame,
                split_metrics,
            ) = run_one(
                args.data_dir,
                args.split_file,
                scenario,
                endpoint,
                args.feature_set,
                args.seed,
                args.max_iter,
            )
            headlines.append(headline)
            all_predictions.append(predictions)
            all_split_metrics.append(split_metrics)
            all_coefficients.append(coef_frame)
            feature_catalog[f"{scenario}:{endpoint}"] = used_features
            print(json.dumps(headline, sort_keys=True))

    summary = pd.DataFrame(headlines).sort_values(["endpoint", "scenario"])
    summary.to_csv(args.output_dir / "patient_endpoint_summary_v2_2.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(
        args.output_dir / "patient_endpoint_predictions_v2_2.csv", index=False
    )
    pd.concat(all_split_metrics, ignore_index=True).to_csv(
        args.output_dir / "patient_endpoint_split_metrics_v2_2.csv", index=False
    )
    pd.concat(all_coefficients, ignore_index=True).to_csv(
        args.output_dir / "patient_endpoint_coefficients_v2_2.csv", index=False
    )
    (args.output_dir / "patient_endpoint_feature_catalog_v2_2.json").write_text(
        json.dumps(
            {
                "feature_set": args.feature_set,
                "seed": args.seed,
                "split_file": str(args.split_file),
                "excluded_by_design": [
                    "all clinical_endpoint columns",
                    "all observation_or_selection columns",
                    "latent unobserved_severity",
                    "other endpoints as features",
                ],
                "model": "LogisticRegression(class_weight='balanced')",
                "loss": (
                    "log-loss / binary cross-entropy with balanced class weights "
                    "(sklearn equivalent of weighted BCE)"
                ),
                "primary_metric": "average_precision",
                "secondary_metrics": ["roc_auc", "brier"],
                "features_by_run": feature_catalog,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
