#!/usr/bin/env python3
"""Summarize Wave 5A path + eval tables across seeds/variants."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=ROOT / "outputs/wave5/eval",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/wave5/WAVE5A_summary.csv",
    )
    args = parser.parse_args()

    frames = []
    for p in sorted(args.eval_dir.glob("eval_*.csv")):
        frames.append(pd.read_csv(p))
    if not frames:
        raise SystemExit(f"No eval_*.csv in {args.eval_dir}")
    df = pd.concat(frames, ignore_index=True)

    metrics = [
        "true_path_mass",
        "path_entropy",
        "path_hhi",
        "path_recall_at_1",
        "path_recall_at_5",
        "path_recall_at_10",
        "mean_true_path_rank",
        "hub_mass",
    ]
    rows = []
    for variant, sub in df.groupby("variant"):
        row = {"variant": variant, "n_query_seed": len(sub)}
        for m in metrics:
            if m not in sub.columns:
                continue
            row[f"{m}_mean"] = float(sub[m].mean())
            row[f"{m}_std"] = float(sub[m].std(ddof=0))
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(
        "true_path_mass_mean", ascending=False
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    print(summary.to_string(index=False))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
