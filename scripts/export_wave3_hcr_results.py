#!/usr/bin/env python3
"""Export WAVE3_HCR runs from local wandb into dated output folder."""

from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WANDB = ROOT / "wandb"


def harvest() -> pd.DataFrame:
    rows = []
    for run_dir in sorted(WANDB.glob("run-*")):
        sp = run_dir / "files" / "wandb-summary.json"
        if not sp.exists():
            continue
        summary = json.loads(sp.read_text())
        blob = ""
        cfg = run_dir / "files" / "config.yaml"
        if cfg.exists():
            blob += cfg.read_text(errors="ignore")
        meta = run_dir / "files" / "wandb-metadata.json"
        if meta.exists():
            blob += meta.read_text(errors="ignore")
        blob += json.dumps(summary)
        if "WAVE3_HCR" not in blob and summary.get("experiment_name") != "WAVE3_HCR":
            continue
        # Skip smoke
        if "WAVE3_HCR_SMOKE" in blob:
            continue

        seed = summary.get("training_seed")
        if seed is None:
            m = re.search(r"training\.seed[=:]\s*(\d+)", blob)
            seed = int(m.group(1)) if m else None

        variant = summary.get("hcr_variant")
        if not variant:
            m = re.search(r"hcr=([a-z_]+)", blob)
            variant = m.group(1) if m else None
        if variant in {None, "none"}:
            m = re.search(r"intervention['\"]?\s*[:=]\s*['\"]?hgt_hcr_([a-z_]+)", blob)
            if m:
                variant = m.group(1)

        rows.append(
            {
                "wave": "wave3_hcr",
                "stage": "WAVE3_HCR",
                "model": "hgt",
                "num_layers": 1,
                "heads": 8,
                "hcr_variant": variant,
                "training_seed": int(seed) if seed is not None else None,
                "valid_auprc": summary.get(
                    "valid_auprc", summary.get("best_valid_AUPRC", summary.get("best_valid_metric"))
                ),
                "valid_auroc": summary.get("valid_auc", summary.get("valid_auroc")),
                "valid_brier": summary.get("valid_brier"),
                "test_auprc": summary.get("test_auprc"),
                "test_auroc": summary.get("test_auc", summary.get("test_auroc")),
                "test_brier": summary.get("test_brier"),
                "parameter_count_trainable": summary.get("parameter_count_trainable"),
                "best_epoch": summary.get("best_epoch"),
                "checkpoint_path": summary.get("checkpoint_path"),
                "run_dir": run_dir.name,
                "mtime": run_dir.stat().st_mtime,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.dropna(subset=["training_seed", "valid_auprc", "hcr_variant"])
    return (
        df.sort_values("mtime")
        .drop_duplicates(["hcr_variant", "training_seed"], keep="first")
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", default=None)
    args = parser.parse_args()
    day = args.day or date.today().isoformat()
    out = ROOT / "outputs" / f"wave3_hcr_{day}"
    out.mkdir(parents=True, exist_ok=True)

    df = harvest()
    if df.empty:
        print("No WAVE3_HCR runs found")
        return
    all_path = out / f"{day}_wave3_HCR_all_runs.csv"
    df.to_csv(all_path, index=False)
    summary = (
        df.groupby("hcr_variant", dropna=False)
        .agg(
            n_seeds=("training_seed", "nunique"),
            mean_valid_auprc=("valid_auprc", "mean"),
            std_valid_auprc=("valid_auprc", "std"),
            mean_test_auprc=("test_auprc", "mean"),
            std_test_auprc=("test_auprc", "std"),
            mean_valid_brier=("valid_brier", "mean"),
            mean_test_brier=("test_brier", "mean"),
            mean_params=("parameter_count_trainable", "mean"),
        )
        .reset_index()
        .sort_values("mean_valid_auprc", ascending=False)
    )
    summary.to_csv(out / f"{day}_wave3_HCR_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {all_path}")


if __name__ == "__main__":
    main()
