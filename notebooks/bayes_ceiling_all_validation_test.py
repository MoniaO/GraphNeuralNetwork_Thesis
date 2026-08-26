"""Compute oracle ceilings for all, validation, and test cohorts.

Ranking ceilings:
  - AUROC and AUPRC from true posterior probabilities p(y=1|x).

Bayes error ceiling (symmetric 0-1 loss):
  - R* = E[min(p, 1-p)] using oracle posteriors from the known DAG.
  - Empirical Bayes error applies the MAP rule (predict 1 iff p >= 0.5).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
WORLD_PATH = Path("/Users/monika/data/synthetic_pharmacotherapy_v3_samples_clean.csv")
NODES_PATH = Path("/Users/monika/data/synthetic_pharmacotherapy_v3_nodes.csv")
EDGES_PATH = Path("/Users/monika/data/synthetic_pharmacotherapy_v3_edges_audited.csv")
SPLITS_PATH = Path("/Users/monika/data/splits/patient_splits_v3.csv")
OUTPUT_PATH = Path("/Users/monika/data/bayes_ceiling_clean_all_validation_test.csv")
SUMMARY_OUTPUT_PATH = Path("/Users/monika/data/bayes_ceiling_summary_by_endpoint.csv")
SUMMARY_LATEX_PATH = Path("/Users/monika/data/bayes_ceiling_summary_by_endpoint.tex")

COHORTS = ("all", "validation", "test")
TARGET_ENDPOINT = None  # E.g. "AKI"; use None for every clinical endpoint.

COL_NODE = "node"
COL_NODE_TYPE = "node_type"
COL_BASE_PREV = "base_prevalence"
COL_SRC = "source"
COL_DST = "target"
COL_EFFECT = "effect_size"
COL_SIGN = "effect_sign"
ENDPOINT_TYPE = "clinical_endpoint"
DEFAULT_EFFECT = 0.5
DEFAULT_SIGN = 1.0
PATIENT_ID = "patient_id"
BAYES_THRESHOLD = 0.5


def logit(p: float | np.ndarray) -> float | np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def read_any(path: Path) -> pd.DataFrame:
    if path.suffix in {".csv", ".gz"}:
        return pd.read_csv(path)
    if path.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file format: {path}")


def roc_auc(y: np.ndarray, scores: np.ndarray) -> float:
    """AUROC via the Mann--Whitney statistic, with average ranks for ties."""
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    start = 0

    while start < len(scores):
        end = start
        while end + 1 < len(scores) and sorted_scores[end + 1] == sorted_scores[start]:
            end += 1
        ranks[order[start:end + 1]] = 0.5 * (start + end) + 1.0
        start = end + 1

    return float(
        (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0)
        / (n_pos * n_neg)
    )


def average_precision(y: np.ndarray, scores: np.ndarray) -> float:
    """Average precision using a step-wise precision--recall definition."""
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if int(y.sum()) == 0:
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y[order]
    true_positives = np.cumsum(y_sorted)
    precision = true_positives / np.arange(1, len(y_sorted) + 1)
    return float((precision * y_sorted).sum() / y_sorted.sum())


def standardize_like_generator(values: np.ndarray) -> np.ndarray:
    """Apply the generator's scaling only to non-binary variables."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]

    if len(finite) == 0 or len(np.unique(finite)) <= 2:
        return values

    sd = float(np.std(finite))
    if sd <= 1e-8:
        return values

    return (values - float(np.mean(finite))) / sd


def build_incoming_index(edges: pd.DataFrame) -> dict[str, list[pd.Series]]:
    incoming: dict[str, list[pd.Series]] = {}
    for _, edge in edges.iterrows():
        incoming.setdefault(str(edge[COL_DST]), []).append(edge)
    return incoming


def oracle_scores(
    data: pd.DataFrame,
    endpoint: str,
    metadata: dict[str, dict[str, object]],
    incoming: dict[str, list[pd.Series]],
) -> tuple[np.ndarray, int, list[str]]:
    base_prevalence = metadata[endpoint].get(COL_BASE_PREV, np.nan)
    if not np.isfinite(base_prevalence):
        raise ValueError(f"Endpoint '{endpoint}' has no finite base_prevalence.")

    eta = np.full(len(data), logit(float(base_prevalence)), dtype=float)
    used_parent_count = 0
    missing_parents: list[str] = []

    for edge in incoming.get(endpoint, []):
        source = str(edge[COL_SRC])
        if source not in data.columns:
            missing_parents.append(source)
            continue

        effect = (
            float(edge[COL_EFFECT])
            if pd.notna(edge[COL_EFFECT])
            else DEFAULT_EFFECT
        )
        sign = (
            float(edge[COL_SIGN])
            if pd.notna(edge[COL_SIGN])
            else DEFAULT_SIGN
        )
        values = standardize_like_generator(data[source].to_numpy(float))
        eta += sign * effect * np.nan_to_num(values)
        used_parent_count += 1

    return sigmoid(eta), used_parent_count, missing_parents


def bayes_error_ceiling(posteriors: np.ndarray) -> float:
    """Theoretical Bayes error: R* = E[min(p, 1 - p)] for symmetric 0-1 loss."""
    p = np.asarray(posteriors, dtype=float)
    return float(np.mean(np.minimum(p, 1.0 - p)))


def bayes_classifier_predictions(
    posteriors: np.ndarray,
    threshold: float = BAYES_THRESHOLD,
) -> np.ndarray:
    """MAP classifier for binary labels: predict 1 iff p >= threshold."""
    p = np.asarray(posteriors, dtype=float)
    return (p >= threshold).astype(int)


def empirical_bayes_error(
    posteriors: np.ndarray,
    labels: np.ndarray,
    threshold: float = BAYES_THRESHOLD,
) -> float:
    """Observed 0-1 loss of the Bayes classifier on a finite sample."""
    predictions = bayes_classifier_predictions(posteriors, threshold=threshold)
    return float(np.mean(predictions != np.asarray(labels, dtype=int)))


def compute_cohort_results(
    data: pd.DataFrame,
    cohort: str,
    endpoints: list[str],
    metadata: dict[str, dict[str, object]],
    incoming: dict[str, list[pd.Series]],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for endpoint in endpoints:
        scores, n_parents_used, missing_parents = oracle_scores(
            data=data,
            endpoint=endpoint,
            metadata=metadata,
            incoming=incoming,
        )
        labels = data[endpoint].to_numpy(int)
        bayes_error = bayes_error_ceiling(scores)
        bayes_error_observed = empirical_bayes_error(scores, labels)

        records.append(
            {
                "cohort": cohort,
                "n_patients": len(data),
                "endpoint": endpoint,
                "n_positive": int(labels.sum()),
                "prevalence": float(labels.mean()),
                "auc_ceiling": roc_auc(labels, scores),
                "auprc_ceiling": average_precision(labels, scores),
                "bayes_error_ceiling": bayes_error,
                "empirical_bayes_error": bayes_error_observed,
                "accuracy_oracle": 1.0 - bayes_error_observed,
                "n_parents_used": n_parents_used,
                "missing_parents": ",".join(missing_parents),
            }
        )

    return pd.DataFrame.from_records(records)


def build_endpoint_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Pivot Bayes error and oracle accuracy to one row per endpoint."""
    subset = results[
        ["endpoint", "cohort", "bayes_error_ceiling", "accuracy_oracle", "prevalence"]
    ].copy()

    wide = subset.pivot_table(
        index="endpoint",
        columns="cohort",
        values=["bayes_error_ceiling", "accuracy_oracle", "prevalence"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}__{cohort}" for metric, cohort in wide.columns]
    wide = wide.reset_index()

    column_order = ["endpoint"]
    for cohort in COHORTS:
        column_order.extend(
            [
                f"prevalence__{cohort}",
                f"bayes_error_ceiling__{cohort}",
                f"accuracy_oracle__{cohort}",
            ]
        )
    existing = [col for col in column_order if col in wide.columns]
    return wide[existing].sort_values("endpoint").reset_index(drop=True)


def summary_to_latex(summary: pd.DataFrame) -> str:
    """Render a booktabs table with Bayes error and oracle accuracy per cohort."""
    rows: list[str] = []
    for _, row in summary.iterrows():
        cells = [str(row["endpoint"]).replace("_", r"\_")]
        for cohort in COHORTS:
            err = row[f"bayes_error_ceiling__{cohort}"]
            acc = row[f"accuracy_oracle__{cohort}"]
            cells.append(f"{err:.4f}")
            cells.append(f"{100.0 * acc:.2f}\\%")
        rows.append(" & ".join(cells) + r" \\")
    header = (
        r"\begin{table}[htbp]" "\n"
        r"  \centering" "\n"
        r"  \caption{Bayes error ceiling and oracle accuracy by clinical endpoint.}" "\n"
        r"  \label{tab:bayes-ceiling-by-endpoint}" "\n"
        r"  \small" "\n"
        r"  \begin{tabular}{lcccccc}" "\n"
        r"    \toprule" "\n"
        r"    & \multicolumn{2}{c}{All} & \multicolumn{2}{c}{Validation} & \multicolumn{2}{c}{Test} \\" "\n"
        r"    Endpoint & Min error & Acc. & Min error & Acc. & Min error & Acc. \\" "\n"
        r"    \midrule" "\n"
    )
    body = "\n".join(f"    {line}" for line in rows)
    footer = (
        r"    \bottomrule" "\n"
        r"  \end{tabular}" "\n"
        r"\end{table}"
    )
    return header + body + "\n" + footer


def main() -> None:
    world = read_any(WORLD_PATH)
    nodes = read_any(NODES_PATH)
    edges = read_any(EDGES_PATH)
    splits = read_any(SPLITS_PATH)

    required_world = {PATIENT_ID}
    required_splits = {PATIENT_ID, "split"}
    missing_world = required_world.difference(world.columns)
    missing_splits = required_splits.difference(splits.columns)

    if missing_world:
        raise ValueError(f"Missing columns in world data: {sorted(missing_world)}")
    if missing_splits:
        raise ValueError(f"Missing columns in split data: {sorted(missing_splits)}")

    metadata = nodes.set_index(COL_NODE).to_dict(orient="index")
    all_endpoints = [
        node
        for node, node_meta in metadata.items()
        if str(node_meta.get(COL_NODE_TYPE, "")) == ENDPOINT_TYPE
        and node in world.columns
    ]

    if not all_endpoints:
        raise ValueError("No clinical endpoints found in the world data.")

    if TARGET_ENDPOINT is None:
        endpoints = all_endpoints
    elif TARGET_ENDPOINT in all_endpoints:
        endpoints = [TARGET_ENDPOINT]
    else:
        raise ValueError(
            f"Unknown endpoint '{TARGET_ENDPOINT}'. Available endpoints: {all_endpoints}"
        )

    split_ids = {
        split: set(
            splits.loc[splits["split"].astype(str) == split, PATIENT_ID]
        )
        for split in ("validation", "test")
    }

    cohorts = {
        "all": world.copy(),
        "validation": world[world[PATIENT_ID].isin(split_ids["validation"])].copy(),
        "test": world[world[PATIENT_ID].isin(split_ids["test"])].copy(),
    }

    for cohort in COHORTS:
        if cohort not in cohorts:
            raise ValueError(f"Unknown cohort requested: {cohort}")
        if cohorts[cohort].empty:
            raise ValueError(f"Cohort '{cohort}' contains no patients.")

    incoming = build_incoming_index(edges)
    results = pd.concat(
        [
            compute_cohort_results(
                data=cohorts[cohort].reset_index(drop=True),
                cohort=cohort,
                endpoints=endpoints,
                metadata=metadata,
                incoming=incoming,
            )
            for cohort in COHORTS
        ],
        ignore_index=True,
    )

    cohort_order = pd.CategoricalDtype(COHORTS, ordered=True)
    results["cohort"] = results["cohort"].astype(cohort_order)
    results = results.sort_values(["endpoint", "cohort"]).reset_index(drop=True)
    results["cohort"] = results["cohort"].astype(str)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    summary = build_endpoint_summary(results)
    summary.to_csv(SUMMARY_OUTPUT_PATH, index=False)
    SUMMARY_LATEX_PATH.write_text(summary_to_latex(summary), encoding="utf-8")

    display_columns = [
        "cohort",
        "endpoint",
        "n_patients",
        "n_positive",
        "prevalence",
        "auc_ceiling",
        "auprc_ceiling",
        "bayes_error_ceiling",
        "empirical_bayes_error",
        "accuracy_oracle",
        "n_parents_used",
        "missing_parents",
    ]

    print("Bayes/oracle ceiling results")
    print("=" * 120)
    print(results[display_columns].to_string(index=False))
    print("=" * 120)
    print(f"Saved: {OUTPUT_PATH}")
    print()
    print("Summary per endpoint (Bayes error ceiling + oracle accuracy)")
    print("=" * 120)
    print(summary.to_string(index=False))
    print("=" * 120)
    print(f"Saved summary CSV: {SUMMARY_OUTPUT_PATH}")
    print(f"Saved summary LaTeX: {SUMMARY_LATEX_PATH}")


if __name__ == "__main__":
    main()
