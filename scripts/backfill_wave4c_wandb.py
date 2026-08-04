#!/usr/bin/env python3
"""Backfill Wave 4C latent-gate results to W&B (they were trained with wandb.enabled=false)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import wandb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "wave4c_latent_2026-07-31"
CSV = OUT / "2026-07-31_wave4c_LATENT_all_runs.csv"
SUMMARY = OUT / "2026-07-31_wave4c_LATENT_summary.csv"

ENTITY = "politechnika-gnn-thesis"
PROJECT = "politechnika-gnn-thesis"
GROUP = "TaskA_WAVE4C_LATENT"


def already_uploaded(api, variant: str, seed: int) -> bool:
    try:
        runs = api.runs(
            f"{ENTITY}/{PROJECT}",
            filters={
                "group": GROUP,
                "config.hcr_variant": variant,
                "config.training_seed": seed,
            },
            per_page=5,
        )
        return any(True for _ in runs)
    except Exception:
        return False


def main() -> None:
    os.environ["WANDB_MODE"] = "online"
    if not CSV.exists():
        raise SystemExit(f"Missing {CSV}")

    df = pd.read_csv(CSV)
    api = wandb.Api()
    uploaded = 0
    skipped = 0

    for _, row in df.iterrows():
        variant = str(row["hcr_variant"])
        seed = int(row["training_seed"])
        metrics_path = OUT / f"latent_metrics_{variant}_{seed}.json"
        metrics = {}
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())

        name = f"wave4c_{variant}_seed{seed}"
        if already_uploaded(api, variant, seed):
            print(f"SKIP already in cloud: {name}")
            skipped += 1
            continue

        run = wandb.init(
            entity=ENTITY,
            project=PROJECT,
            group=GROUP,
            job_type="backfill",
            name=name,
            tags=["TaskA", "WAVE4C", "LATENT", "backfill", variant],
            config={
                "wave": "WAVE4C_LATENT",
                "hcr_variant": variant,
                "training_seed": seed,
                "experiment": "WAVE4C_LATENT",
                "motif_completion.hide": "parent_a",
                "latent_gate.enabled": True,
                "backfill": True,
                "source_csv": str(CSV),
                "checkpoint_path": row.get("checkpoint_path"),
            },
            reinit=True,
        )
        payload = {
            "valid/auprc": float(row["valid_auprc"]) if pd.notna(row["valid_auprc"]) else None,
            "valid/brier": float(row["valid_brier"]) if pd.notna(row["valid_brier"]) else None,
            "test/auprc": float(row["test_auprc"]) if pd.notna(row["test_auprc"]) else None,
            "test/brier": float(row["test_brier"]) if pd.notna(row["test_brier"]) else None,
            "best_epoch": int(row["best_epoch"]) if pd.notna(row["best_epoch"]) else None,
            "motif/auprc_heldout_vs_valid_neg": float(row["auprc_motif"])
            if pd.notna(row["auprc_motif"])
            else None,
            "motif/mean_prob": float(row["mean_prob_motif"])
            if pd.notna(row["mean_prob_motif"])
            else None,
            "motif/n_positives": int(row["n_motif_positives"])
            if pd.notna(row["n_motif_positives"])
            else None,
        }
        # Flatten selected fields from metrics JSON.
        if metrics:
            payload["motif/n_edges_found"] = metrics.get("n_motif_edges_found")
            payload["motif/n_positives_json"] = metrics.get("n_motif_positives")
        payload = {k: v for k, v in payload.items() if v is not None}
        wandb.log(payload)
        for k, v in payload.items():
            wandb.run.summary[k] = v
        wandb.run.summary["protocol"] = "Wave 4C latent-gate HCR; hide A→G; features (A,Y,B)"
        print(f"UPLOADED {name} → {run.url}")
        wandb.finish()
        uploaded += 1

    # Also log aggregate summary table as a tiny run.
    if SUMMARY.exists():
        summary = pd.read_csv(SUMMARY)
        run = wandb.init(
            entity=ENTITY,
            project=PROJECT,
            group=GROUP,
            job_type="backfill_summary",
            name="wave4c_LATENT_summary",
            tags=["TaskA", "WAVE4C", "LATENT", "summary", "backfill"],
            config={"wave": "WAVE4C_LATENT", "backfill_summary": True},
            reinit=True,
        )
        wandb.log({"wave4c_summary": wandb.Table(dataframe=summary)})
        for _, r in summary.iterrows():
            wandb.run.summary[f"mean_auprc/{r['variant']}"] = float(r["mean"])
        print(f"UPLOADED summary table → {run.url}")
        wandb.finish()

    print(f"\nDone. uploaded={uploaded} skipped={skipped}")


if __name__ == "__main__":
    main()
