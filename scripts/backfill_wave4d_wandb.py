#!/usr/bin/env python3
"""Backfill completed Wave 4D metrics to W&B (covers offline-trained runs)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import wandb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "wave4d_context_role_audit_2026-07-31"
CSV = OUT / "2026-07-31_wave4d_all_runs.csv"

ENTITY = "politechnika-gnn-thesis"
PROJECT = "politechnika-gnn-thesis"
GROUP = "TaskA_WAVE4D_CONTEXT_ROLE_AUDIT"


def already_uploaded(api, variant: str, seed: int, hide: str) -> bool:
    try:
        runs = api.runs(
            f"{ENTITY}/{PROJECT}",
            filters={
                "group": GROUP,
                "config.hcr_variant": variant,
                "config.training_seed": seed,
                "config.hide": hide,
            },
            per_page=5,
        )
        return any(True for _ in runs)
    except Exception:
        return False


def main() -> None:
    os.environ["WANDB_MODE"] = "online"
    files = sorted(OUT.glob("wave4d_metrics_*.json"))
    if not files:
        raise SystemExit(f"No metrics in {OUT}")

    api = wandb.Api()
    uploaded = skipped = 0

    for path in files:
        m = json.loads(path.read_text())
        variant = str(m["hcr"])
        seed = int(m["seed"])
        hide = str(m["hide"])
        name = f"wave4d_{variant}_{hide}_seed{seed}"
        if already_uploaded(api, variant, seed, hide):
            print(f"SKIP {name}")
            skipped += 1
            continue

        pa = m.get("panel_a") or {}
        pb = m.get("panel_b") or {}
        run = wandb.init(
            entity=ENTITY,
            project=PROJECT,
            group=GROUP,
            job_type="backfill",
            name=name,
            tags=["TaskA", "WAVE4D", "backfill", variant, hide],
            config={
                "wave": "WAVE4D_CONTEXT_ROLE_AUDIT",
                "hcr_variant": variant,
                "training_seed": seed,
                "hide": hide,
                "backfill": True,
                "metrics_path": str(path),
            },
            reinit=True,
        )
        payload = {
            "motif/auprc": pa.get("auprc_motif"),
            "motif/mrr": pa.get("mrr"),
            "motif/hits_at_1": pa.get("hits_at_1"),
            "motif/hits_at_3": pa.get("hits_at_3"),
            "motif/hits_at_5": pa.get("hits_at_5"),
            "motif/mean_rank": pa.get("mean_rank"),
            "motif/median_rank": pa.get("median_rank"),
            "context/selection_precision_at_1": m.get("context_selection_precision_at_1"),
            "role/fpr_confounder": (pb.get("confounder") or {}).get("fpr_at_0.5"),
            "role/fpr_mediator": (pb.get("mediator") or {}).get("fpr_at_0.5"),
            "role/fpr_collider": (pb.get("collider") or {}).get("fpr_at_0.5"),
            "role/fpr_descendant": (pb.get("descendant") or {}).get("fpr_at_0.5"),
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        wandb.log(payload)
        for k, v in payload.items():
            wandb.run.summary[k] = v
        if m.get("selected_z_role_counts"):
            wandb.run.summary["selected_z_role_counts"] = m["selected_z_role_counts"]
        print(f"UPLOADED {name} → {run.url}")
        wandb.finish()
        uploaded += 1

    if CSV.exists():
        df = pd.read_csv(CSV)
        run = wandb.init(
            entity=ENTITY,
            project=PROJECT,
            group=GROUP,
            job_type="backfill_summary",
            name="wave4d_all_runs_table",
            tags=["TaskA", "WAVE4D", "summary", "backfill"],
            config={"wave": "WAVE4D_CONTEXT_ROLE_AUDIT", "backfill_summary": True},
            reinit=True,
        )
        wandb.log({"wave4d_all_runs": wandb.Table(dataframe=df)})
        print(f"UPLOADED all_runs table → {run.url}")
        wandb.finish()

    print(f"\nDone. uploaded={uploaded} skipped={skipped}")


if __name__ == "__main__":
    main()
