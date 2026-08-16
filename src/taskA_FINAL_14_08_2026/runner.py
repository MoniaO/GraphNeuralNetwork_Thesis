"""FINAL 14.08.2026 runner — S10 × {mlp,kan} × 6 scenarios × 5 seeds."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import torch

from taskA_FINAL_14_08_2026 import (
    BLOCK_ID,
    CANDIDATE_SEED,
    FINAL_SEEDS,
    FROZEN,
    MODEL_NAME,
    SCENARIOS,
)
from taskA_final_large_grid_11_08_2026.stage_c.variants import get_variant

REPO = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO / "outputs" / "taskA_FINAL_14.08.2026"
RUNS = OUT_ROOT / "runs"
LOG_ROOT = OUT_ROOT / "logs"
PROGRESS_EVERY = 5

EncoderKind = Literal["mlp", "kan_shallow"]


def run_dir_for(
    *,
    encoder: EncoderKind,
    scenario: str,
    seed: int,
) -> Path:
    return (
        RUNS
        / encoder
        / scenario
        / "S10_HCR_FULL40"
        / f"seed{int(seed)}"
    )


def build_overrides(
    *,
    encoder: EncoderKind,
    scenario: str,
    training_seed: int,
    epochs: int | None = None,
    patience: int | None = None,
) -> list[str]:
    variant = get_variant("S10")
    raw_dim = max(int(variant.raw_dim), 1)
    epochs = int(epochs if epochs is not None else FROZEN["epochs"])
    patience = int(patience if patience is not None else FROZEN["patience"])
    enc_tag = "mlp" if encoder == "mlp" else "kan"
    tags = (
        f"[TaskA,FINAL_14.08.2026,S10,{enc_tag},"
        f"hgt_L2_h32_hd4,hcr_none]"
    )
    return [
        "model=TaskA_hgt_final",
        "hcr=none",
        "data=dataset_v3",
        f"data.dataset.scenario={scenario}",
        f"data.candidate_seed={CANDIDATE_SEED}",
        f"training.seed={int(training_seed)}",
        f"training.lr={FROZEN['lr']}",
        f"training.weight_decay={FROZEN['weight_decay']}",
        f"training.epochs={epochs}",
        f"training.early_stopping_patience={patience}",
        f"training.grad_clip={FROZEN['grad_clip']}",
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
        f"model.heads={FROZEN['heads']}",
        f"model.hgt.heads={FROZEN['heads']}",
        f"model.hgt.hidden_dim={FROZEN['hidden_dim']}",
        f"model.hgt.num_layers={FROZEN['n_layers']}",
        f"model.hgt.dropout={FROZEN['dropout']}",
        "model.decoder.name=fusion88_stat",
        "model.decoder.dropout=0.2",
        f"++model.decoder.stat_raw_dim={raw_dim}",
        "++model.decoder.force_zero_stat=false",
        "++model.decoder.pair_dropout=0.1",
        f"++model.decoder.stat_pair_encoder={encoder}",
        "++model.decoder.spline_l1=1.0e-5",
        "wandb.enabled=true",
        "wandb.project=politechnika-gnn-thesis",
        "wandb.entity=politechnika-gnn-thesis",
        "wandb.group=TaskA_FINAL_14_08_2026",
        f"wandb.job_type=final_{enc_tag}",
        f"wandb.tags={tags}",
        (
            f"++wandb.run_name=FINAL__{enc_tag}__{scenario}"
            f"__S10__seed{int(training_seed)}"
        ),
        f"experiment.wave={BLOCK_ID}",
        f"experiment.variant={MODEL_NAME}_{enc_tag.upper()}",
        f"++experiment.stat_variant={variant.config_id}",
        f"++experiment.stat_pair_encoder={encoder}",
    ]


def extract_result(out_dir: Path, *, exit_code: int) -> dict[str, Any]:
    ckpt = out_dir / "best_model.pt"
    result: dict[str, Any] = {
        "exit_code": exit_code,
        "out_dir": str(out_dir),
        "checkpoint": str(ckpt) if ckpt.exists() else None,
        "status": "ok" if exit_code == 0 and ckpt.exists() else "failed",
        "model_name": MODEL_NAME,
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
        "note": "SEALED — do not use for FINAL selection",
    }
    result["candidate_seed"] = blob.get("candidate_seed")
    result["candidate_fingerprint"] = blob.get("candidate_fingerprint")
    if int(result.get("candidate_seed") or -1) != int(CANDIDATE_SEED):
        result["status"] = "STOP_candidate_seed_mismatch"
    return result


def launch_one(
    *,
    encoder: EncoderKind,
    scenario: str,
    training_seed: int,
    epochs: int | None = None,
    patience: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    out_dir = run_dir_for(encoder=encoder, scenario=scenario, seed=training_seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    overrides = build_overrides(
        encoder=encoder,
        scenario=scenario,
        training_seed=training_seed,
        epochs=epochs,
        patience=patience,
    )
    overrides.append(f"hydra.run.dir={out_dir}")
    overrides.append("hydra.job.chdir=false")
    cmd = [sys.executable, str(REPO / "src" / "train_taskA.py"), *overrides]
    meta = {
        "cmd": cmd,
        "model_name": MODEL_NAME,
        "encoder": encoder,
        "variant": "S10_HCR_FULL40",
        "scenario": scenario,
        "training_seed": training_seed,
        "candidate_seed": CANDIDATE_SEED,
        "frozen": FROZEN,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "launch_meta_14.08.2026.json").write_text(
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

    log_path = out_dir / "train_14.08.2026.log"
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
    result["encoder"] = encoder
    result["scenario"] = scenario
    result["training_seed"] = training_seed
    (out_dir / "result_14.08.2026.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def _flush(rows: list[dict[str, Any]], *, i: int, n_jobs: int, final: bool = False) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    ok = sum(1 for r in rows if r.get("status") == "ok")
    fail = sum(1 for r in rows if r.get("status") not in {"ok", None, "dry_run"})
    status = {
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "done": len(rows),
        "total": n_jobs,
        "ok": ok,
        "fail": fail,
        "model_name": MODEL_NAME,
        "final": final,
    }
    (LOG_ROOT / "FINAL_STATUS_14.08.2026.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    (OUT_ROOT / "FINAL_SUMMARY_PARTIAL_14.08.2026.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print(f"[flush] i={i}/{n_jobs} ok={ok} fail={fail}", flush=True)


def run_final_grid(
    *,
    encoders: tuple[EncoderKind, ...] = ("mlp", "kan_shallow"),
    scenarios: tuple[str, ...] = SCENARIOS,
    seeds: tuple[int, ...] = FINAL_SEEDS,
    epochs: int | None = None,
    patience: int | None = None,
    dry_run: bool = False,
    max_jobs: int | None = None,
) -> Path:
    jobs: list[tuple[EncoderKind, str, int]] = []
    for enc in encoders:
        for sc in scenarios:
            for seed in seeds:
                jobs.append((enc, sc, seed))
                if max_jobs is not None and len(jobs) >= max_jobs:
                    break
            if max_jobs is not None and len(jobs) >= max_jobs:
                break
        if max_jobs is not None and len(jobs) >= max_jobs:
            break

    rows: list[dict[str, Any]] = []
    n = len(jobs)
    for i, (enc, sc, seed) in enumerate(jobs, 1):
        out_dir = run_dir_for(encoder=enc, scenario=sc, seed=seed)
        prior = out_dir / "result_14.08.2026.json"
        if prior.exists() and not dry_run:
            try:
                prev = json.loads(prior.read_text(encoding="utf-8"))
            except Exception:
                prev = {}
            if prev.get("status") == "ok":
                print(f"[{i}/{n}] SKIP ok {enc} {sc} seed={seed}", flush=True)
                row = {
                    "encoder": enc,
                    "scenario": sc,
                    "training_seed": seed,
                    "status": "ok",
                    "skipped_existing": True,
                }
                if "selection" in prev:
                    row.update(prev["selection"])
                rows.append(row)
                if i % PROGRESS_EVERY == 0:
                    _flush(rows, i=i, n_jobs=n)
                continue

        print(f"[{i}/{n}] {enc} {sc} seed={seed}", flush=True)
        res = launch_one(
            encoder=enc,
            scenario=sc,
            training_seed=seed,
            epochs=epochs,
            patience=patience,
            dry_run=dry_run,
        )
        row = {
            "encoder": enc,
            "scenario": sc,
            "training_seed": seed,
            "status": res.get("status"),
        }
        if "selection" in res:
            row.update(res["selection"])
        rows.append(row)
        if i % PROGRESS_EVERY == 0:
            _flush(rows, i=i, n_jobs=n)

    _flush(rows, i=n, n_jobs=n, final=True)
    summary = OUT_ROOT / "FINAL_SUMMARY_14.08.2026.json"
    summary.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {summary}", flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="FINAL 14.08.2026 S10 MLP+KAN grid")
    p.add_argument("--mode", choices=["count", "smoke", "full"], default="count")
    p.add_argument(
        "--encoder",
        action="append",
        choices=["mlp", "kan_shallow"],
        default=None,
        help="Repeatable; default both mlp and kan_shallow",
    )
    p.add_argument("--scenario", action="append", default=None)
    p.add_argument("--max-jobs", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    encs: tuple[EncoderKind, ...]
    if args.encoder:
        encs = tuple(args.encoder)  # type: ignore[assignment]
    else:
        encs = ("mlp", "kan_shallow")
    scens = tuple(args.scenario) if args.scenario else SCENARIOS

    if args.mode == "count":
        print(f"encoders={encs} scenarios={len(scens)} seeds={len(FINAL_SEEDS)}")
        print(f"total_jobs={len(encs) * len(scens) * len(FINAL_SEEDS)}")
        return 0

    if args.mode == "smoke":
        run_final_grid(
            encoders=encs,
            scenarios=(scens[0],),
            seeds=(FINAL_SEEDS[0],),
            epochs=args.epochs or 1,
            patience=args.patience or 1,
            dry_run=args.dry_run,
            max_jobs=args.max_jobs or 2,
        )
        return 0

    run_final_grid(
        encoders=encs,
        scenarios=scens,
        seeds=FINAL_SEEDS,
        epochs=args.epochs,
        patience=args.patience,
        dry_run=args.dry_run,
        max_jobs=args.max_jobs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
