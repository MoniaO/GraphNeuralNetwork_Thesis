#!/usr/bin/env python3
"""04 — (opcjonalnie) krzywe Stage A → W&B. Nie trenuje od nowa."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import wandb

REPO = Path(__file__).resolve().parents[2]
ROOT = (
    REPO
    / "outputs"
    / "taskA_final_large_grid_11.08.2026"
    / "stage_a"
    / "runs"
    / "shared"
)

EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)\s+\|\s+optim\s+([0-9.nan]+)\s+\|\s+"
    r"train AUPRC\s+([0-9.nan]+)\s+\|\s+"
    r"valid AUPRC\s+([0-9.nan]+)\s+\|\s+"
    r"valid AUC\s+([0-9.nan]+)"
)


def parse_log(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = EPOCH_RE.search(line)
        if not m:
            continue
        ep, loss, tr, va, auc = m.groups()
        rows.append(
            {
                "epoch": int(ep),
                "train/optim_loss": float(loss),
                "train/auprc": float(tr),
                "valid/auprc": float(va),
                "valid/auc": float(auc),
            }
        )
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="clean")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    base = ROOT / args.scenario
    results = sorted(base.rglob("result_11.08.2026.json"))
    n = 0
    for res_path in results:
        d = json.loads(res_path.read_text(encoding="utf-8"))
        if d.get("status") != "ok":
            continue
        log = res_path.parent / "train_11.08.2026.log"
        if not log.exists():
            continue
        curves = parse_log(log)
        if not curves:
            continue
        cfg_id = res_path.parent.parent.name
        seed = res_path.parent.name
        sel = d.get("selection") or {}
        name = f"BACKFILL_STAGE_A__{args.scenario}__{cfg_id}__{seed}"
        print(f"upload {name}  epochs={len(curves)}  best_valid_auprc={sel.get('valid_auprc')}")
        if args.dry_run:
            n += 1
            if args.limit and n >= args.limit:
                break
            continue
        run = wandb.init(
            project="politechnika-gnn-thesis",
            entity="politechnika-gnn-thesis",
            group="TaskA_FINAL_LARGE_GRID_11_08_2026_STAGE_A",
            job_type="stage_a_backfill_curves",
            name=name,
            tags=[
                "TaskA",
                "FINAL_LARGE_GRID",
                "11.08.2026",
                "STAGE_A",
                "backfill",
                "backbone_race",
                "no_hcr",
                args.scenario,
            ],
            config={
                "config_id": cfg_id,
                "seed": seed,
                "scenario": args.scenario,
                "candidate_seed": d.get("candidate_seed"),
                "candidate_fingerprint": d.get("candidate_fingerprint"),
                "source_result": str(res_path),
            },
            reinit=True,
        )
        for row in curves:
            wandb.log(
                {
                    "train/optim_loss": row["train/optim_loss"],
                    "train/auprc": row["train/auprc"],
                    "valid/auprc": row["valid/auprc"],
                    "valid/auc": row["valid/auc"],
                },
                step=row["epoch"],
            )
        wandb.summary["best_valid_AUPRC"] = sel.get("valid_auprc")
        wandb.summary["best_valid_AUC"] = sel.get("valid_auc")
        wandb.summary["best_valid_brier"] = sel.get("valid_brier")
        wandb.summary["best_epoch"] = sel.get("best_epoch")
        sealed = d.get("sealed_test") or {}
        wandb.summary["sealed_test_auprc"] = sealed.get("test_auprc")
        wandb.summary["sealed_test_auc"] = sealed.get("test_auc")
        wandb.summary["loss_first"] = curves[0]["train/optim_loss"]
        wandb.summary["loss_last"] = curves[-1]["train/optim_loss"]
        wandb.summary["loss_min"] = min(r["train/optim_loss"] for r in curves)
        run.finish()
        n += 1
        if args.limit and n >= args.limit:
            break
    print(f"uploaded={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
