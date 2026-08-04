#!/usr/bin/env python3
"""Decide H16 vs H8 for wave-2 HGT L1 from paired valid AUPRC.

Rule (as specified for closing wave 2):
  Δ_valid = AUPRC_H16 − AUPRC_H8
  Choose H16 only if mean(Δ_valid) > 0; otherwise keep locked H8.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def latest_final_csv() -> Path:
    finals = sorted(PROJECT_ROOT.glob("outputs/wave2_architecture_*_FINAL/*_ARCH_FINAL_all_runs.csv"))
    if not finals:
        raise FileNotFoundError("No ARCH_FINAL all_runs.csv found under outputs/")
    return finals[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", default=None)
    args = parser.parse_args()
    day = args.day or date.today().isoformat()

    out_dir = PROJECT_ROOT / "outputs" / f"wave2_architecture_{day}_HEADS_BOUNDARY"
    h16_path = out_dir / f"{day}_wave2_ARCH_HEADS_BOUNDARY_all_runs.csv"
    if not h16_path.exists():
        raise FileNotFoundError(h16_path)

    h16 = pd.read_csv(h16_path)
    h16 = h16[(h16["model"].astype(str) == "hgt") & (h16["num_layers"].astype(int) == 1) & (h16["heads"].astype(int) == 16)]
    h16 = h16[["training_seed", "valid_auprc", "test_auprc", "valid_brier"]].rename(
        columns={
            "valid_auprc": "valid_h16",
            "test_auprc": "test_h16",
            "valid_brier": "brier_h16",
        }
    )

    final_csv = latest_final_csv()
    h8 = pd.read_csv(final_csv)
    h8 = h8[(h8["model"].astype(str) == "hgt") & (h8["num_layers"].astype(int) == 1) & (h8["heads"].astype(float) == 8)]
    h8 = h8[["training_seed", "valid_auprc", "test_auprc", "valid_brier"]].rename(
        columns={
            "valid_auprc": "valid_h8",
            "test_auprc": "test_h8",
            "valid_brier": "brier_h8",
        }
    )

    cmp = h16.merge(h8, on="training_seed", how="inner").sort_values("training_seed")
    if cmp.empty:
        raise RuntimeError("No paired seeds between H16 BOUNDARY and FINAL H8")

    cmp["delta_valid_H16_minus_H8"] = cmp["valid_h16"] - cmp["valid_h8"]
    cmp["delta_test_H16_minus_H8"] = cmp["test_h16"] - cmp["test_h8"]
    cmp_path = out_dir / f"{day}_wave2_ARCH_HEADS_BOUNDARY_H16_vs_H8.csv"
    cmp.to_csv(cmp_path, index=False)

    mean_d = float(cmp["delta_valid_H16_minus_H8"].mean())
    std_d = float(cmp["delta_valid_H16_minus_H8"].std(ddof=1)) if len(cmp) > 1 else float("nan")
    wins = int((cmp["delta_valid_H16_minus_H8"] > 0).sum())
    n = len(cmp)
    choose_h16 = bool(mean_d > 0)
    locked = "hgt_L1_H16" if choose_h16 else "hgt_L1_H8"

    decision = pd.DataFrame(
        [
            {
                "mean_delta_valid_H16_minus_H8": mean_d,
                "std_delta_valid": std_d,
                "wins_valid": wins,
                "n_seeds": n,
                "mean_valid_H16": float(cmp["valid_h16"].mean()),
                "mean_valid_H8": float(cmp["valid_h8"].mean()),
                "mean_test_H16": float(cmp["test_h16"].mean()),
                "mean_test_H8": float(cmp["test_h8"].mean()),
                "choose_H16": choose_h16,
                "locked_config": locked,
                "rule": "Choose H16 only if mean(Δ valid AUPRC H16−H8) > 0; else keep H8.",
                "h8_source": str(final_csv.relative_to(PROJECT_ROOT)),
            }
        ]
    )
    decision_path = out_dir / f"{day}_wave2_ARCH_HEADS_BOUNDARY_decision.csv"
    decision.to_csv(decision_path, index=False)

    manifest = out_dir / "MANIFEST.txt"
    manifest.write_text(
        "\n".join(
            [
                "Wave: 2 — ARCH_HEADS_BOUNDARY (H16 vs H8)",
                f"Date: {day}",
                "Config: HGT L1 hidden=64 heads=16 dropout=0.2 residual=true",
                f"H16 runs: {len(h16)} | paired vs FINAL H8: {n}",
                f"mean Δ valid (H16−H8): {mean_d:+.4f} ± {std_d:.4f}",
                f"wins: {wins}/{n}",
                f"DECISION: {locked} (choose_H16={choose_h16})",
                f"Compare: {cmp_path.name}",
                f"Decision: {decision_path.name}",
                "Closes wave 2 architecture selection.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(manifest.read_text())
    print(f"Wrote {cmp_path}")
    print(f"Wrote {decision_path}")


if __name__ == "__main__":
    main()
