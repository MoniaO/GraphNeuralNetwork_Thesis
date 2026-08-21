"""Stage C statistical screen runner (11.08.2026).

Frozen backbone: HGT h32 L2 d0.25 lr1e-3 heads=4
(heads=8 refine lost: 0.705 vs 0.729 — do not retune).

Variants S0–S10 × scenarios × 3 seeds. Selection = valid AUPRC only.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from taskA.experiments.stage_a_backbone.grid import (
    CANDIDATE_SEED,
    FIXED,
    SCENARIOS,
    SCREENING_SEEDS,
)
from taskA.features.variants import (
    STAGE_C_VARIANTS,
    StageCVariant,
    get_variant,
)

REPO = Path(__file__).resolve().parents[4]
OUT_ROOT = REPO / "outputs" / "taskA_final_large_grid_11.08.2026" / "stage_c"
LOG_ROOT = REPO / "outputs" / "taskA_final_large_grid_11.08.2026" / "logs"
PROGRESS_EVERY = 10

# Frozen HGT Top-1 (Stage A); heads=4 won refine vs 8.
FROZEN = {
    "backbone": "hgt",
    "hidden_dim": 32,
    "n_layers": 2,
    "dropout": 0.25,
    "lr": 1e-3,
    "heads": 4,
}


def _flush(
    summary_rows: list[dict[str, Any]],
    *,
    i: int,
    n_jobs: int,
    scenario: str,
    heads: int,
    final: bool = False,
) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    ok = sum(1 for r in summary_rows if r.get("status") == "ok")
    fail = sum(1 for r in summary_rows if r.get("status") not in {"ok", None})
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    status = {
        "updated": stamp,
        "done": len(summary_rows),
        "total": n_jobs,
        "ok": ok,
        "fail": fail,
        "scenario": scenario,
        "heads": heads,
        "phase": "stage_c_stats_screen",
        "frozen_backbone": FROZEN,
        "wandb": {
            "enabled": True,
            "project": "politechnika-gnn-thesis",
            "entity": "politechnika-gnn-thesis",
            "group": "TaskA_FINAL_LARGE_GRID_11_08_2026_STAGE_C",
        },
        "final": final,
        "note": (
            f"Per-run: stage_c/runs/.../result_11.08.2026.json; "
            f"flush every {PROGRESS_EVERY}. heads=8 refine lost — default heads=4."
        ),
    }
    (LOG_ROOT / "STAGE_C_STATUS_11.08.2026.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    summary_path = OUT_ROOT / f"STAGE_C_SUMMARY_{scenario}_hd{heads}_11.08.2026.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    chk = OUT_ROOT / "progress_every10_11.08.2026"
    chk.mkdir(parents=True, exist_ok=True)
    tag = "FINAL" if final else f"{i:03d}"
    (chk / f"checkpoint_{scenario}_hd{heads}_{tag}_11.08.2026.json").write_text(
        json.dumps({"status": status, "rows": summary_rows}, indent=2),
        encoding="utf-8",
    )
    print(
        f"[flush] i={i}/{n_jobs} ok={ok} fail={fail} → {summary_path.name}"
    )


def build_overrides(
    variant: StageCVariant,
    *,
    scenario: str,
    training_seed: int,
    heads: int = 4,
    epochs: int | None = None,
    patience: int | None = None,
) -> list[str]:
    epochs = int(epochs if epochs is not None else FIXED["epochs"])
    patience = int(patience if patience is not None else FIXED["patience"])
    raw_dim = max(int(variant.raw_dim), 1)
    force_zero = variant.id == "S0"
    tags = (
        f"[TaskA,FINAL_LARGE_GRID,11.08.2026,STAGE_C,"
        f"{variant.config_id},heads{int(heads)},hcr_none]"
    )
    ov = [
        "model=hgt_fusion88",
        "hcr=none",
        "data=dataset_v3",
        f"data.dataset.scenario={scenario}",
        f"data.candidate_seed={CANDIDATE_SEED}",
        f"training.seed={int(training_seed)}",
        f"training.lr={FROZEN['lr']}",
        f"training.weight_decay={FIXED['weight_decay']}",
        f"training.epochs={epochs}",
        f"training.early_stopping_patience={patience}",
        f"training.grad_clip={FIXED['grad_clip']}",
        "training.selection_metric=auprc",
        "training.device=cpu",
        f"model.name={FROZEN['backbone']}",
        f"model.conv_type={FROZEN['backbone']}",
        f"model.encoder_name={FROZEN['backbone']}",
        f"model.hidden_dim={FROZEN['hidden_dim']}",
        f"model.hidden_channels={FROZEN['hidden_dim']}",
        f"model.num_layers={FROZEN['n_layers']}",
        f"model.dropout={FROZEN['dropout']}",
        "model.residual=true",
        f"model.heads={int(heads)}",
        f"model.hgt.heads={int(heads)}",
        f"model.hgt.hidden_dim={FROZEN['hidden_dim']}",
        f"model.hgt.num_layers={FROZEN['n_layers']}",
        f"model.hgt.dropout={FROZEN['dropout']}",
        "model.decoder.name=fusion88_stat",
        "model.decoder.dropout=0.2",
        f"++model.decoder.stat_raw_dim={raw_dim}",
        f"++model.decoder.force_zero_stat={str(force_zero).lower()}",
        "++model.decoder.pair_dropout=0.1",
        "wandb.enabled=true",
        "wandb.project=politechnika-gnn-thesis",
        "wandb.entity=politechnika-gnn-thesis",
        "wandb.group=TaskA_FINAL_LARGE_GRID_11_08_2026_STAGE_C",
        f"wandb.job_type=stage_c_{variant.id}",
        f"wandb.tags={tags}",
        (
            f"++wandb.run_name=STAGE_C__{scenario}__{variant.config_id}"
            f"__hd{int(heads)}__seed{int(training_seed)}"
        ),
        "experiment.wave=FINAL_LARGE_GRID_11_08_2026",
        f"experiment.variant=STAGE_C_{variant.config_id}",
        f"++experiment.stat_variant={variant.config_id}",
    ]
    return ov


def run_dir_for(
    variant: StageCVariant, scenario: str, seed: int, *, heads: int
) -> Path:
    return (
        OUT_ROOT
        / "runs"
        / scenario
        / f"hd{int(heads)}"
        / variant.config_id
        / f"seed{seed}"
    )


def extract_result(out_dir: Path, *, exit_code: int) -> dict[str, Any]:
    ckpt = out_dir / "best_model.pt"
    result: dict[str, Any] = {
        "exit_code": exit_code,
        "out_dir": str(out_dir),
        "checkpoint": str(ckpt) if ckpt.exists() else None,
        "status": "ok" if exit_code == 0 and ckpt.exists() else "failed",
    }
    if not ckpt.exists():
        return result
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    valid = blob.get("final_valid_metrics") or {}
    test = blob.get("final_test_metrics") or {}
    result["selection"] = {
        "best_valid_metric": float(blob.get("best_valid_metric", float("nan"))),
        "best_valid_metric_name": blob.get("best_valid_metric_name"),
        "best_epoch": blob.get("best_epoch"),
        "valid_auprc": float(valid.get("auprc", float("nan"))),
        "valid_auc": float(valid.get("auc", float("nan"))),
        "valid_brier": float(valid.get("brier", float("nan"))),
    }
    result["sealed_test"] = {
        "test_auprc": float(test.get("auprc", float("nan"))),
        "test_auc": float(test.get("auc", float("nan"))),
        "test_brier": float(test.get("brier", float("nan"))),
        "note": "SEALED — do not use for Stage C selection",
    }
    result["candidate_seed"] = blob.get("candidate_seed")
    result["candidate_fingerprint"] = blob.get("candidate_fingerprint")
    if int(result.get("candidate_seed") or -1) != int(CANDIDATE_SEED):
        result["status"] = "STOP_candidate_seed_mismatch"
    return result


def launch_one(
    variant: StageCVariant,
    *,
    scenario: str,
    training_seed: int,
    heads: int = 4,
    epochs: int | None = None,
    patience: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    import subprocess

    out_dir = run_dir_for(variant, scenario, training_seed, heads=heads)
    out_dir.mkdir(parents=True, exist_ok=True)
    overrides = build_overrides(
        variant,
        scenario=scenario,
        training_seed=training_seed,
        heads=heads,
        epochs=epochs,
        patience=patience,
    )
    overrides.append(f"hydra.run.dir={out_dir}")
    overrides.append("hydra.job.chdir=false")
    cmd = [sys.executable, str(REPO / "src" / "taskA" / "training" / "train.py"), *overrides]
    meta = {
        "cmd": cmd,
        "variant": variant.config_id,
        "raw_dim": variant.raw_dim,
        "scenario": scenario,
        "training_seed": training_seed,
        "candidate_seed": CANDIDATE_SEED,
        "heads": heads,
        "frozen": FROZEN,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "launch_meta_11.08.2026.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    if dry_run:
        return {"status": "dry_run", "out_dir": str(out_dir), "cmd": cmd}

    env = os.environ.copy()
    env.setdefault(
        "GSN_PROJECT_ROOT",
        str(Path.home() / "Desktop" / "GSN Graphs dysertation 2026"),
    )
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")

    log_path = out_dir / "train_11.08.2026.log"
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False,
        )
    result = extract_result(out_dir, exit_code=proc.returncode)
    result["variant"] = variant.config_id
    result["heads"] = heads
    (out_dir / "result_11.08.2026.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def run_stage_c_screen(
    *,
    scenario: str = "clean",
    variants: tuple[str, ...] | None = None,
    seeds: tuple[int, ...] = SCREENING_SEEDS,
    heads: int = 4,
    epochs: int | None = None,
    patience: int | None = None,
    dry_run: bool = False,
    max_jobs: int | None = None,
) -> Path:
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}; expected one of {SCENARIOS}")
    if int(heads) not in {4, 8}:
        raise ValueError("heads must be 4 (default) or 8 (optional; lost refine)")

    var_list: list[StageCVariant]
    if variants is None:
        var_list = list(STAGE_C_VARIANTS)
    else:
        var_list = [get_variant(v) for v in variants]

    jobs: list[tuple[StageCVariant, int]] = []
    for v in var_list:
        for seed in seeds:
            jobs.append((v, seed))
            if max_jobs is not None and len(jobs) >= max_jobs:
                break
        if max_jobs is not None and len(jobs) >= max_jobs:
            break

    summary_rows: list[dict[str, Any]] = []
    n_jobs = len(jobs)
    for i, (variant, seed) in enumerate(jobs, 1):
        out_dir = run_dir_for(variant, scenario, seed, heads=heads)
        prior = out_dir / "result_11.08.2026.json"
        if prior.exists() and not dry_run:
            try:
                prev = json.loads(prior.read_text(encoding="utf-8"))
            except Exception:
                prev = {}
            if prev.get("status") == "ok":
                print(f"[{i}/{n_jobs}] SKIP ok {variant.config_id} seed={seed}")
                row = {
                    "variant": variant.config_id,
                    "variant_id": variant.id,
                    "raw_dim": variant.raw_dim,
                    "scenario": scenario,
                    "training_seed": seed,
                    "heads": heads,
                    "status": "ok",
                    "skipped_existing": True,
                }
                if "selection" in prev:
                    row.update(prev["selection"])
                    row["candidate_fingerprint"] = prev.get("candidate_fingerprint")
                summary_rows.append(row)
                if i % PROGRESS_EVERY == 0:
                    _flush(
                        summary_rows,
                        i=i,
                        n_jobs=n_jobs,
                        scenario=scenario,
                        heads=heads,
                    )
                continue

        print(
            f"[{i}/{n_jobs}] {variant.config_id} seed={seed} "
            f"scenario={scenario} heads={heads}"
        )
        res = launch_one(
            variant,
            scenario=scenario,
            training_seed=seed,
            heads=heads,
            epochs=epochs,
            patience=patience,
            dry_run=dry_run,
        )
        row = {
            "variant": variant.config_id,
            "variant_id": variant.id,
            "raw_dim": variant.raw_dim,
            "scenario": scenario,
            "training_seed": seed,
            "heads": heads,
            "status": res.get("status"),
        }
        if "selection" in res:
            row.update(res["selection"])
            row["candidate_fingerprint"] = res.get("candidate_fingerprint")
        summary_rows.append(row)
        if i % PROGRESS_EVERY == 0:
            _flush(
                summary_rows, i=i, n_jobs=n_jobs, scenario=scenario, heads=heads
            )

    _flush(
        summary_rows,
        i=n_jobs,
        n_jobs=n_jobs,
        scenario=scenario,
        heads=heads,
        final=True,
    )
    summary_path = OUT_ROOT / f"STAGE_C_SUMMARY_{scenario}_hd{heads}_11.08.2026.json"
    print(f"Wrote {summary_path}")
    return summary_path


def run_stage_c_multi_scenario(
    *,
    scenarios: tuple[str, ...] = SCENARIOS,
    variants: tuple[str, ...] = ("S9", "S10"),
    seeds: tuple[int, ...] = SCREENING_SEEDS,
    heads: int = 4,
    epochs: int | None = None,
    patience: int | None = None,
    dry_run: bool = False,
) -> Path:
    """Run selected variants on multiple scenarios (skip existing ok)."""
    all_rows: list[dict[str, Any]] = []
    for sc in scenarios:
        path = run_stage_c_screen(
            scenario=sc,
            variants=variants,
            seeds=seeds,
            heads=heads,
            epochs=epochs,
            patience=patience,
            dry_run=dry_run,
        )
        try:
            all_rows.extend(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            pass
    multi_path = (
        OUT_ROOT
        / f"STAGE_C_MULTI_{'_'.join(variants)}_hd{heads}_11.08.2026.json"
    )
    multi_path.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    print(f"Wrote {multi_path} rows={len(all_rows)}")
    return multi_path


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Stage C statistical screen 11.08.2026")
    p.add_argument(
        "--mode",
        choices=["smoke", "screen", "count", "multi"],
        default="smoke",
    )
    p.add_argument("--scenario", default="clean")
    p.add_argument(
        "--all-scenarios",
        action="store_true",
        help="With --mode multi: run all 6 GSN scenarios",
    )
    p.add_argument(
        "--variant",
        action="append",
        default=None,
        help="Variant id(s) e.g. S9 S10; default all S0–S10 (screen) or S9+S10 (multi)",
    )
    p.add_argument("--heads", type=int, default=4, help="Default 4 (won refine); 8 optional")
    p.add_argument("--max-jobs", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    if args.mode == "count":
        n_var = len(args.variant) if args.variant else len(STAGE_C_VARIANTS)
        n_sc = len(SCENARIOS) if args.all_scenarios or args.mode == "multi" else 1
        print(f"variants × scenarios × 3 seeds = {n_var * n_sc * 3}")
        return 0

    if args.mode == "smoke":
        vars_ = args.variant or ["S0", "S1"]
        run_stage_c_screen(
            scenario=args.scenario,
            variants=tuple(vars_),
            seeds=(SCREENING_SEEDS[0],),
            heads=int(args.heads),
            epochs=args.epochs or 1,
            patience=args.patience or 1,
            dry_run=args.dry_run,
        )
        return 0

    if args.mode == "multi":
        vars_ = tuple(args.variant) if args.variant else ("S9", "S10")
        scens = SCENARIOS if args.all_scenarios else (args.scenario,)
        run_stage_c_multi_scenario(
            scenarios=scens,
            variants=vars_,
            heads=int(args.heads),
            epochs=args.epochs,
            patience=args.patience,
            dry_run=args.dry_run,
        )
        return 0

    run_stage_c_screen(
        scenario=args.scenario,
        variants=tuple(args.variant) if args.variant else None,
        heads=int(args.heads),
        epochs=args.epochs,
        patience=args.patience,
        dry_run=args.dry_run,
        max_jobs=args.max_jobs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
