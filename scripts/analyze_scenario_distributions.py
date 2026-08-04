#!/usr/bin/env python3
"""Compare train-patient distributions across GSN v3 scenarios and hospitals."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcr.variable_specs_v3 import VARIABLE_SPECS  # noqa: E402


SCENARIOS = (
    "clean",
    "hidden_confounder",
    "selection_bias",
    "no_overlap",
    "noisy_documentation",
    "multihospital",
)
ENDPOINTS = (
    "AKI",
    "DILI",
    "Depression",
    "Falls",
    "Delirium",
    "GI_bleeding",
    "Hyponatremia",
    "Hyperkalemia",
    "QT_arrhythmia",
    "Hospitalization",
    "Serotonin_syndrome",
    "Rhabdomyolysis",
    "Lactic_acidosis",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            os.environ.get(
                "GSN_PROJECT_ROOT",
                Path.home() / "Desktop" / "GSN Graphs dysertation 2026",
            )
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs"
        / "Wave 0-11 podsumowanie"
        / "scenario_distribution_audit",
    )
    return parser.parse_args()


def effective_column(frame: pd.DataFrame, scenario: str, variable: str) -> str | None:
    recorded = f"recorded_{variable}"
    if scenario == "noisy_documentation" and recorded in frame.columns:
        return recorded
    if variable in frame.columns:
        return variable
    return None


def load_train_scenarios(data_root: Path) -> dict[str, pd.DataFrame]:
    data_dir = data_root / "2 v3. Data" / "dataset_v3"
    split_path = data_root / "2 v3. Data" / "splits" / "patient_splits_v3.csv"
    splits = pd.read_csv(split_path, usecols=["patient_id", "split"])
    train_ids = set(
        splits.loc[splits["split"].astype(str).eq("train"), "patient_id"].astype(int)
    )
    frames: dict[str, pd.DataFrame] = {}
    for scenario in SCENARIOS:
        path = data_dir / f"synthetic_pharmacotherapy_v3_samples_{scenario}.csv"
        frame = pd.read_csv(path)
        frame = frame[frame["patient_id"].astype(int).isin(train_ids)].copy()
        frames[scenario] = frame
    return frames


def numeric_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=float)


def describe_vector(values: np.ndarray) -> dict[str, float | int]:
    if values.size == 0:
        return {
            "n_complete": 0,
            "mean": math.nan,
            "std": math.nan,
            "median": math.nan,
            "q05": math.nan,
            "q25": math.nan,
            "q75": math.nan,
            "q95": math.nan,
        }
    q05, q25, median, q75, q95 = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "n_complete": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "median": float(median),
        "q05": float(q05),
        "q25": float(q25),
        "q75": float(q75),
        "q95": float(q95),
    }


def distribution_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []
    for scenario, frame in frames.items():
        for variable, spec in VARIABLE_SPECS.items():
            if variable == "unobserved_severity":
                continue
            column = effective_column(frame, scenario, spec.column_name)
            if column is None:
                continue
            values = numeric_values(frame, column)
            desc = describe_vector(values)
            rows.append(
                {
                    "scenario": scenario,
                    "variable": variable,
                    "effective_column": column,
                    "variable_type": spec.variable_type.value,
                    "n_patients": len(frame),
                    "missing_rate": float(
                        pd.to_numeric(frame[column], errors="coerce").isna().mean()
                    ),
                    "prevalence": (
                        float(np.mean(values)) if spec.variable_type.value == "binary" else math.nan
                    ),
                    **desc,
                }
            )
    return pd.DataFrame(rows)


def bernoulli_js(p: float, q: float) -> float:
    eps = 1e-12
    p = float(np.clip(p, eps, 1.0 - eps))
    q = float(np.clip(q, eps, 1.0 - eps))
    m = 0.5 * (p + q)

    def kl(a: float, b: float) -> float:
        return a * math.log(a / b) + (1.0 - a) * math.log((1.0 - a) / (1.0 - b))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def shift_rows(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    *,
    reference_name: str,
    comparison_name: str,
    scenario_for_columns: str,
) -> list[dict]:
    rows: list[dict] = []
    for variable, spec in VARIABLE_SPECS.items():
        if variable == "unobserved_severity":
            continue
        ref_col = effective_column(reference, "clean", spec.column_name)
        cmp_col = effective_column(comparison, scenario_for_columns, spec.column_name)
        if ref_col is None or cmp_col is None:
            continue
        ref = numeric_values(reference, ref_col)
        cmp = numeric_values(comparison, cmp_col)
        if ref.size < 2 or cmp.size < 2:
            continue
        ref_mean, cmp_mean = float(ref.mean()), float(cmp.mean())
        ref_sd, cmp_sd = float(ref.std(ddof=0)), float(cmp.std(ddof=0))
        pooled_sd = math.sqrt((ref_sd**2 + cmp_sd**2) / 2.0)
        scale = max(ref_sd, 1e-8)
        row = {
            "reference": reference_name,
            "comparison": comparison_name,
            "variable": variable,
            "variable_type": spec.variable_type.value,
            "reference_mean": ref_mean,
            "comparison_mean": cmp_mean,
            "mean_delta": cmp_mean - ref_mean,
            "reference_missing_rate": float(
                pd.to_numeric(reference[ref_col], errors="coerce").isna().mean()
            ),
            "comparison_missing_rate": float(
                pd.to_numeric(comparison[cmp_col], errors="coerce").isna().mean()
            ),
            "missing_rate_delta": float(
                pd.to_numeric(comparison[cmp_col], errors="coerce").isna().mean()
                - pd.to_numeric(reference[ref_col], errors="coerce").isna().mean()
            ),
            "standardized_mean_difference": (
                (cmp_mean - ref_mean) / pooled_sd if pooled_sd > 1e-8 else 0.0
            ),
            "wasserstein_distance": float(wasserstein_distance(ref, cmp)),
            "wasserstein_over_clean_sd": float(wasserstein_distance(ref, cmp) / scale),
            "ks_statistic": float(ks_2samp(ref, cmp, method="asymp").statistic),
            "prevalence_delta": math.nan,
            "bernoulli_js_divergence": math.nan,
        }
        if spec.variable_type.value == "binary":
            row["prevalence_delta"] = cmp_mean - ref_mean
            row["bernoulli_js_divergence"] = bernoulli_js(ref_mean, cmp_mean)
        rows.append(row)
    return rows


def scenario_shifts(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    clean = frames["clean"]
    rows: list[dict] = []
    for scenario in SCENARIOS:
        if scenario == "clean":
            continue
        rows.extend(
            shift_rows(
                clean,
                frames[scenario],
                reference_name="clean",
                comparison_name=scenario,
                scenario_for_columns=scenario,
            )
        )
    shifts = pd.DataFrame(rows)
    shifts["absolute_effect"] = np.where(
        shifts["variable_type"].eq("binary"),
        shifts["prevalence_delta"].abs(),
        shifts["standardized_mean_difference"].abs(),
    )
    return shifts


def aggregate_shifts(shifts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for comparison, group in shifts.groupby("comparison", sort=False):
        binary = group[group["variable_type"].eq("binary")]
        numeric = group[~group["variable_type"].eq("binary")]
        rows.append(
            {
                "scenario": comparison,
                "n_variables": len(group),
                "median_abs_binary_prevalence_delta": float(
                    binary["prevalence_delta"].abs().median()
                ),
                "p90_abs_binary_prevalence_delta": float(
                    binary["prevalence_delta"].abs().quantile(0.9)
                ),
                "max_abs_binary_prevalence_delta": float(
                    binary["prevalence_delta"].abs().max()
                ),
                "median_abs_numeric_smd": float(
                    numeric["standardized_mean_difference"].abs().median()
                ),
                "p90_abs_numeric_smd": float(
                    numeric["standardized_mean_difference"].abs().quantile(0.9)
                ),
                "max_abs_numeric_smd": float(
                    numeric["standardized_mean_difference"].abs().max()
                ),
                "median_ks_statistic": float(group["ks_statistic"].median()),
                "p90_ks_statistic": float(group["ks_statistic"].quantile(0.9)),
                "mean_abs_missing_rate_delta": float(
                    group["missing_rate_delta"].abs().mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def endpoint_prevalence(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []
    for scenario, frame in frames.items():
        for endpoint in ENDPOINTS:
            column = effective_column(frame, scenario, endpoint)
            if column is None:
                continue
            values = numeric_values(frame, column)
            rows.append(
                {
                    "scope": "scenario",
                    "group": scenario,
                    "endpoint": endpoint,
                    "n_patients": len(frame),
                    "prevalence": float(values.mean()),
                }
            )
    mh = frames["multihospital"]
    for hospital, group in mh.groupby("hospital_id", sort=True):
        for endpoint in ENDPOINTS:
            if endpoint not in group.columns:
                continue
            values = numeric_values(group, endpoint)
            rows.append(
                {
                    "scope": "hospital",
                    "group": str(hospital),
                    "endpoint": endpoint,
                    "n_patients": len(group),
                    "prevalence": float(values.mean()),
                }
            )
    return pd.DataFrame(rows)


def hospital_distributions(multihospital: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for hospital, frame in multihospital.groupby("hospital_id", sort=True):
        for variable, spec in VARIABLE_SPECS.items():
            if variable == "unobserved_severity" or spec.column_name not in frame.columns:
                continue
            values = numeric_values(frame, spec.column_name)
            desc = describe_vector(values)
            rows.append(
                {
                    "hospital_id": str(hospital),
                    "variable": variable,
                    "variable_type": spec.variable_type.value,
                    "n_patients": len(frame),
                    "missing_rate": float(
                        pd.to_numeric(frame[spec.column_name], errors="coerce").isna().mean()
                    ),
                    "prevalence": (
                        float(values.mean()) if spec.variable_type.value == "binary" else math.nan
                    ),
                    **desc,
                }
            )
    return pd.DataFrame(rows)


def hospital_shifts(multihospital: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for hospital, comparison in multihospital.groupby("hospital_id", sort=True):
        reference = multihospital[multihospital["hospital_id"] != hospital]
        rows.extend(
            shift_rows(
                reference,
                comparison,
                reference_name="other_hospitals",
                comparison_name=str(hospital),
                scenario_for_columns="multihospital",
            )
        )
    shifts = pd.DataFrame(rows)
    shifts["absolute_effect"] = np.where(
        shifts["variable_type"].eq("binary"),
        shifts["prevalence_delta"].abs(),
        shifts["standardized_mean_difference"].abs(),
    )
    return shifts


def hospital_summary(multihospital: pd.DataFrame, shifts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for hospital, group in shifts.groupby("comparison", sort=True):
        binary = group[group["variable_type"].eq("binary")]
        numeric = group[~group["variable_type"].eq("binary")]
        rows.append(
            {
                "hospital_id": hospital,
                "n_train_patients": int(
                    (multihospital["hospital_id"].astype(str) == hospital).sum()
                ),
                "median_abs_binary_prevalence_delta_vs_others": float(
                    binary["prevalence_delta"].abs().median()
                ),
                "p90_abs_binary_prevalence_delta_vs_others": float(
                    binary["prevalence_delta"].abs().quantile(0.9)
                ),
                "median_abs_numeric_smd_vs_others": float(
                    numeric["standardized_mean_difference"].abs().median()
                ),
                "p90_abs_numeric_smd_vs_others": float(
                    numeric["standardized_mean_difference"].abs().quantile(0.9)
                ),
                "median_ks_statistic_vs_others": float(group["ks_statistic"].median()),
                "p90_ks_statistic_vs_others": float(group["ks_statistic"].quantile(0.9)),
            }
        )
    return pd.DataFrame(rows)


def top_effects(shifts: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    return (
        shifts.sort_values(["comparison", "absolute_effect"], ascending=[True, False])
        .groupby("comparison", sort=False)
        .head(n)
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames = load_train_scenarios(args.data_root)

    distributions = distribution_table(frames)
    shifts = scenario_shifts(frames)
    scenario_summary = aggregate_shifts(shifts)
    endpoints = endpoint_prevalence(frames)
    mh_distributions = hospital_distributions(frames["multihospital"])
    mh_shifts = hospital_shifts(frames["multihospital"])
    mh_summary = hospital_summary(frames["multihospital"], mh_shifts)

    outputs = {
        "scenario_variable_distributions.csv": distributions,
        "scenario_shifts_vs_clean.csv": shifts,
        "scenario_shift_summary.csv": scenario_summary,
        "scenario_top20_shifts.csv": top_effects(shifts),
        "endpoint_prevalence_scenarios_and_hospitals.csv": endpoints,
        "multihospital_variable_distributions.csv": mh_distributions,
        "multihospital_shifts_vs_other_hospitals.csv": mh_shifts,
        "multihospital_shift_summary.csv": mh_summary,
        "multihospital_top20_shifts.csv": top_effects(mh_shifts),
    }
    for name, table in outputs.items():
        table.to_csv(args.out_dir / name, index=False)

    manifest = {
        "scope": "train patients only",
        "scenarios": list(SCENARIOS),
        "scenario_n_train": {key: int(len(value)) for key, value in frames.items()},
        "hospital_n_train": {
            str(key): int(value)
            for key, value in frames["multihospital"]["hospital_id"].value_counts().sort_index().items()
        },
        "variable_registry_size_excluding_latent": int(
            len(VARIABLE_SPECS) - int("unobserved_severity" in VARIABLE_SPECS)
        ),
        "binary_shift": "absolute prevalence difference; Bernoulli JS divergence",
        "numeric_shift": "standardized mean difference, KS statistic, Wasserstein distance",
        "hospital_reference": "each hospital compared with pooled other hospitals",
        "no_p_values": True,
        "outputs": list(outputs),
    }
    (args.out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
