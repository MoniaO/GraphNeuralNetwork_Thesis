#!/usr/bin/env python3
"""Stage-2 decision: binary_compact (5 seeds) vs frozen HGT L1 H8 FINAL."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def latest_final_hgt() -> pd.DataFrame:
    finals = sorted(ROOT.glob("outputs/wave2_architecture_*_FINAL/*_ARCH_FINAL_all_runs.csv"))
    if not finals:
        raise FileNotFoundError("No ARCH_FINAL CSV found")
    df = pd.read_csv(finals[-1])
    h8 = df[(df["model"] == "hgt") & (df["num_layers"] == 1) & (df["heads"].astype(float) == 8)].copy()
    return h8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", default=None)
    args = parser.parse_args()
    day = args.day or date.today().isoformat()
    out = ROOT / "outputs" / f"wave3_hcr_{day}"
    hcr_path = out / f"{day}_wave3_HCR_all_runs.csv"
    if not hcr_path.exists():
        raise FileNotFoundError(hcr_path)

    hcr = pd.read_csv(hcr_path)
    compact = hcr[hcr["hcr_variant"].astype(str) == "binary_compact"].copy()
    baseline = latest_final_hgt()
    base = baseline[
        ["training_seed", "valid_auprc", "test_auprc", "valid_brier", "test_brier", "parameter_count_trainable"]
    ].rename(
        columns={
            "valid_auprc": "valid_baseline",
            "test_auprc": "test_baseline",
            "valid_brier": "brier_valid_baseline",
            "test_brier": "brier_test_baseline",
            "parameter_count_trainable": "params_baseline",
        }
    )

    cmp = compact.merge(base, on="training_seed", how="inner").sort_values("training_seed")
    if cmp.empty:
        raise RuntimeError("No paired seeds between compact and FINAL H8")

    cmp["delta_valid"] = cmp["valid_auprc"] - cmp["valid_baseline"]
    cmp["delta_test"] = cmp["test_auprc"] - cmp["test_baseline"]
    cmp["delta_brier_valid"] = cmp["valid_brier"] - cmp["brier_valid_baseline"]

    paired_path = out / f"{day}_wave3_HCR_STAGE2_paired_deltas.csv"
    cmp.to_csv(paired_path, index=False)

    d = cmp["delta_valid"].to_numpy(dtype=float)
    n = len(cmp)
    wins = int((d > 0).sum())
    mean_d = float(np.mean(d))
    std_d = float(np.std(d, ddof=1)) if n > 1 else float("nan")

    # Success criteria from protocol
    ok_mean = mean_d >= 0.05
    ok_wins = wins >= 4 if n >= 5 else wins == n
    ok_var = std_d <= 0.05 if n > 1 else True
    ok_brier = float(cmp["delta_brier_valid"].mean()) <= 0.01  # not clearly worse
    ok_test = float(cmp["delta_test"].mean()) > 0
    confirmed = bool(ok_mean and ok_wins and ok_var and ok_brier and ok_test)

    table = pd.DataFrame(
        [
            {
                "variant": "HGT_L1_H8",
                "n_seeds": n,
                "valid_auprc": float(cmp["valid_baseline"].mean()),
                "std_valid": float(cmp["valid_baseline"].std(ddof=1)) if n > 1 else float("nan"),
                "test_auprc": float(cmp["test_baseline"].mean()),
                "valid_brier": float(cmp["brier_valid_baseline"].mean()),
                "test_brier": float(cmp["brier_test_baseline"].mean()),
                "wins_vs_other": n - wins,
                "params": float(cmp["params_baseline"].mean()),
            },
            {
                "variant": "HGT_L1_H8+binary_compact",
                "n_seeds": n,
                "valid_auprc": float(cmp["valid_auprc"].mean()),
                "std_valid": float(cmp["valid_auprc"].std(ddof=1)) if n > 1 else float("nan"),
                "test_auprc": float(cmp["test_auprc"].mean()),
                "valid_brier": float(cmp["valid_brier"].mean()),
                "test_brier": float(cmp["test_brier"].mean()),
                "wins_vs_other": wins,
                "params": float(cmp["parameter_count_trainable"].mean()),
            },
        ]
    )
    table_path = out / f"{day}_wave3_HCR_STAGE2_table.csv"
    table.to_csv(table_path, index=False)

    decision = pd.DataFrame(
        [
            {
                "n_seeds": n,
                "mean_delta_valid": mean_d,
                "std_delta_valid": std_d,
                "min_delta_valid": float(np.min(d)),
                "max_delta_valid": float(np.max(d)),
                "wins_valid": wins,
                "mean_delta_test": float(cmp["delta_test"].mean()),
                "mean_delta_brier_valid": float(cmp["delta_brier_valid"].mean()),
                "criterion_mean_ge_0.05": ok_mean,
                "criterion_wins_ge_4_of_5": ok_wins,
                "criterion_std_le_0.05": ok_var,
                "criterion_brier_not_worse": ok_brier,
                "criterion_test_positive": ok_test,
                "stage2_confirmed": confirmed,
            }
        ]
    )
    decision_path = out / f"{day}_wave3_HCR_STAGE2_decision.csv"
    decision.to_csv(decision_path, index=False)

    manifest = out / "MANIFEST_STAGE2.txt"
    manifest.write_text(
        "\n".join(
            [
                "Wave 3 — HCR Stage 2 (binary_compact × 5 seeds)",
                f"Date: {day}",
                f"Paired seeds vs FINAL HGT L1 H8: {n}",
                f"mean Δ valid: {mean_d:+.4f} ± {std_d:.4f}",
                f"min/max Δ valid: {float(np.min(d)):+.4f} / {float(np.max(d)):+.4f}",
                f"wins: {wins}/{n}",
                f"mean Δ test: {float(cmp['delta_test'].mean()):+.4f}",
                f"mean Δ valid Brier: {float(cmp['delta_brier_valid'].mean()):+.4f}",
                f"STAGE2 CONFIRMED: {confirmed}",
                f"Table: {table_path.name}",
                f"Paired: {paired_path.name}",
                f"Decision: {decision_path.name}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(manifest.read_text())
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
