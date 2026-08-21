"""Stage A backbone race runner (11.08.2026).

Hard rules enforced here:
- candidate_seed fixed at 20260722 (never = training seed unless coincident)
- hcr disabled; decoder=fusion88 → g_stat=zeros(24)
- selection uses validation AUPRC only (test metrics sealed, not ranked)
"""

from __future__ import annotations

import json
import os
import subprocess
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
    StageAConfig,
    count_shared_jobs,
    iter_refine_grid,
    iter_shared_grid,
)

REPO = Path(__file__).resolve().parents[4]
OUT_ROOT = REPO / "outputs" / "taskA_final_large_grid_11.08.2026" / "stage_a"
LOG_ROOT = REPO / "outputs" / "taskA_final_large_grid_11.08.2026" / "logs"
PROGRESS_EVERY = 10


def _flush_outputs(
    summary_rows: list[dict[str, Any]],
    *,
    i: int,
    n_jobs: int,
    scenario: str,
    final: bool = False,
) -> None:
    """Write progress artifacts to outputs/ every PROGRESS_EVERY jobs (+ final)."""
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
        "phase": "stage_a_shared_screen",
        "wandb": {
            "enabled": True,
            "project": "politechnika-gnn-thesis",
            "entity": "politechnika-gnn-thesis",
            "group": "TaskA_FINAL_LARGE_GRID_11_08_2026_STAGE_A",
        },
        "final": final,
        "note": (
            "Per-run: stage_a/runs/.../result_11.08.2026.json + W&B epoch curves; "
            f"aggregate flush every {PROGRESS_EVERY} jobs."
        ),
    }
    (LOG_ROOT / "STAGE_A_STATUS_11.08.2026.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    summary_path = OUT_ROOT / "STAGE_A_SHARED_SCREEN_SUMMARY_11.08.2026.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    chk_dir = OUT_ROOT / "progress_every10_11.08.2026"
    chk_dir.mkdir(parents=True, exist_ok=True)
    tag = "FINAL" if final else f"{i:03d}"
    (chk_dir / f"checkpoint_{tag}_11.08.2026.json").write_text(
        json.dumps(
            {"status": status, "rows": summary_rows},
            indent=2,
        ),
        encoding="utf-8",
    )
    # Refresh decision ranking tables (valid AUPRC only).
    try:
        subprocess.run(
            [
                sys.executable,
                str(REPO / "scripts" / "summarize_stage_a_backbone_11.08.2026.py"),
                "--scenario",
                scenario,
                "--total",
                str(n_jobs),
            ],
            cwd=str(REPO),
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:  # noqa: BLE001 — progress must not kill the race
        print(f"[warn] summarize failed: {exc}")
    print(
        f"[flush] i={i}/{n_jobs} ok={ok} fail={fail} → {summary_path.name} "
        f"+ progress_every10/checkpoint_{tag}_11.08.2026.json"
    )


def build_overrides(
    cfg: StageAConfig,
    *,
    scenario: str,
    training_seed: int,
    epochs: int | None = None,
    patience: int | None = None,
) -> list[str]:
    if int(training_seed) == int(CANDIDATE_SEED):
        # Allowed only as coincidence of the fixed lists; still set explicitly apart.
        pass
    epochs = int(epochs if epochs is not None else FIXED["epochs"])
    patience = int(patience if patience is not None else FIXED["patience"])

    # hcr=none is mandatory for Stage A (statistical evidence is Stage C / FINAL).
    ov = [
        "model=hgt_fusion88",  # base group; architecture fields overridden below
        "hcr=none",
        "data=dataset_v3",
        f"data.dataset.scenario={scenario}",
        f"data.candidate_seed={CANDIDATE_SEED}",
        f"training.seed={int(training_seed)}",
        f"training.lr={cfg.lr}",
        f"training.weight_decay={FIXED['weight_decay']}",
        f"training.epochs={epochs}",
        f"training.early_stopping_patience={patience}",
        f"training.grad_clip={FIXED['grad_clip']}",
        "training.selection_metric=auprc",
        "training.device=cpu",
        f"model.name={cfg.backbone}",
        f"model.conv_type={cfg.backbone}",
        f"model.encoder_name={cfg.backbone}",
        f"model.hidden_dim={cfg.hidden_dim}",
        f"model.hidden_channels={cfg.hidden_dim}",
        f"model.num_layers={cfg.n_layers}",
        f"model.dropout={cfg.dropout}",
        "model.residual=true",
        "model.decoder.name=fusion88",
        "model.decoder.dropout=0.2",
        # Neutralize nested model.hgt.* from hgt_fusion88.yaml so it cannot
        # silently pin hidden/layers to the Wave7C defaults.
        f"model.hgt.hidden_dim={cfg.hidden_dim}",
        f"model.hgt.num_layers={cfg.n_layers}",
        f"model.hgt.dropout={cfg.dropout}",
        # Weights & Biases — full epoch curves (optim/loss, AUPRC, AUC)
        "wandb.enabled=true",
        "wandb.project=politechnika-gnn-thesis",
        "wandb.entity=politechnika-gnn-thesis",
        "wandb.group=TaskA_FINAL_LARGE_GRID_11_08_2026_STAGE_A",
        f"wandb.job_type=stage_a_{cfg.phase}",
        f"wandb.tags=[TaskA,FINAL_LARGE_GRID,11.08.2026,STAGE_A,{cfg.phase},no_hcr,{cfg.backbone}]",
        (
            f"+wandb.run_name=STAGE_A__{scenario}__{cfg.backbone}"
            f"__h{cfg.hidden_dim}__L{cfg.n_layers}__d{cfg.dropout:g}__lr{cfg.lr:g}"
            + (f"__hd{int(cfg.heads)}" if cfg.heads is not None else "")
            + (f"__b{cfg.num_bases}" if cfg.num_bases is not None else "")
            + f"__{cfg.phase}__seed{int(training_seed)}"
        ),
        "experiment.wave=FINAL_LARGE_GRID_11_08_2026",
        f"experiment.variant=STAGE_A_{cfg.backbone.upper()}_{cfg.phase.upper()}",
    ]
    if cfg.heads is not None:
        ov.append(f"model.heads={int(cfg.heads)}")
        ov.append(f"model.hgt.heads={int(cfg.heads)}")
    if cfg.num_bases is not None:
        # May be absent from base YAML struct — use + to append.
        ov.append(f"+model.num_bases={cfg.num_bases}")
    return ov


def frozen_hgt_top1() -> StageAConfig:
    """User freeze: HGT Top-1 shared winner (L2, heads=4)."""
    return StageAConfig(
        backbone="hgt",
        hidden_dim=32,
        n_layers=2,
        dropout=0.25,
        lr=1e-3,
        heads=4,
        phase="shared",
    )


def run_dir_for(
    cfg: StageAConfig, scenario: str, seed: int
) -> Path:
    return (
        OUT_ROOT
        / "runs"
        / cfg.phase
        / scenario
        / cfg.config_id
        / f"seed{seed}"
    )


def launch_one(
    cfg: StageAConfig,
    *,
    scenario: str,
    training_seed: int,
    epochs: int | None = None,
    patience: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    out_dir = run_dir_for(cfg, scenario, training_seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    overrides = build_overrides(
        cfg,
        scenario=scenario,
        training_seed=training_seed,
        epochs=epochs,
        patience=patience,
    )
    # Force Hydra output into our dated block.
    overrides.append(f"hydra.run.dir={out_dir}")
    overrides.append("hydra.job.chdir=false")

    cmd = [
        sys.executable,
        str(REPO / "src" / "taskA" / "training" / "train.py"),
        *overrides,
    ]
    meta = {
        "cmd": cmd,
        "config": cfg.to_dict(),
        "scenario": scenario,
        "training_seed": training_seed,
        "candidate_seed": CANDIDATE_SEED,
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
    (out_dir / "result_11.08.2026.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


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
    # SELECTION fields only from valid / best_valid
    result["selection"] = {
        "best_valid_metric": float(blob.get("best_valid_metric", float("nan"))),
        "best_valid_metric_name": blob.get("best_valid_metric_name"),
        "best_epoch": blob.get("best_epoch"),
        "valid_auprc": float(valid.get("auprc", float("nan"))),
        "valid_auc": float(valid.get("auc", float("nan"))),
        "valid_brier": float(valid.get("brier", float("nan"))),
    }
    # Sealed — never used for Stage A ranking
    result["sealed_test"] = {
        "test_auprc": float(test.get("auprc", float("nan"))),
        "test_auc": float(test.get("auc", float("nan"))),
        "test_brier": float(test.get("brier", float("nan"))),
        "note": "SEALED — do not use for model selection before Stage C freeze",
    }
    result["candidate_seed"] = blob.get("candidate_seed")
    result["candidate_fingerprint"] = blob.get("candidate_fingerprint")
    if int(result["candidate_seed"]) != int(CANDIDATE_SEED):
        result["status"] = "STOP_candidate_seed_mismatch"
    return result


def run_shared_screen(
    *,
    scenario: str = "clean",
    seeds: tuple[int, ...] = SCREENING_SEEDS,
    max_jobs: int | None = None,
    epochs: int | None = None,
    patience: int | None = None,
    dry_run: bool = False,
    backbones: tuple[str, ...] | None = None,
) -> Path:
    jobs = []
    for cfg in iter_shared_grid():
        if backbones is not None and cfg.backbone not in backbones:
            continue
        for seed in seeds:
            jobs.append((cfg, seed))
            if max_jobs is not None and len(jobs) >= max_jobs:
                break
        if max_jobs is not None and len(jobs) >= max_jobs:
            break

    summary_rows: list[dict[str, Any]] = []
    n_jobs = len(jobs)
    for i, (cfg, seed) in enumerate(jobs, 1):
        out_dir = run_dir_for(cfg, scenario, seed)
        prior = out_dir / "result_11.08.2026.json"
        if prior.exists() and not dry_run:
            try:
                prev = json.loads(prior.read_text(encoding="utf-8"))
            except Exception:
                prev = {}
            if prev.get("status") == "ok":
                print(
                    f"[{i}/{n_jobs}] SKIP ok {cfg.config_id} seed={seed}"
                )
                row = {
                    "config_id": cfg.config_id,
                    **cfg.to_dict(),
                    "scenario": scenario,
                    "training_seed": seed,
                    "status": "ok",
                    "skipped_existing": True,
                }
                if "selection" in prev:
                    row.update(prev["selection"])
                    row["candidate_fingerprint"] = prev.get(
                        "candidate_fingerprint"
                    )
                summary_rows.append(row)
                if i % PROGRESS_EVERY == 0:
                    _flush_outputs(
                        summary_rows, i=i, n_jobs=n_jobs, scenario=scenario
                    )
                continue

        print(f"[{i}/{n_jobs}] {cfg.config_id} seed={seed} scenario={scenario}")
        res = launch_one(
            cfg,
            scenario=scenario,
            training_seed=seed,
            epochs=epochs,
            patience=patience,
            dry_run=dry_run,
        )
        row = {
            "config_id": cfg.config_id,
            **cfg.to_dict(),
            "scenario": scenario,
            "training_seed": seed,
            "status": res.get("status"),
        }
        if "selection" in res:
            row.update(res["selection"])
            row["candidate_fingerprint"] = res.get("candidate_fingerprint")
        summary_rows.append(row)

        if i % PROGRESS_EVERY == 0:
            _flush_outputs(summary_rows, i=i, n_jobs=n_jobs, scenario=scenario)

    summary_path = OUT_ROOT / "STAGE_A_SHARED_SCREEN_SUMMARY_11.08.2026.json"
    _flush_outputs(
        summary_rows, i=n_jobs, n_jobs=n_jobs, scenario=scenario, final=True
    )
    print(f"Wrote {summary_path}")
    print(f"Shared grid size (all backbones × 3 seeds): {count_shared_jobs(3)}")
    return summary_path


def run_hgt_heads_refine(
    *,
    scenario: str = "clean",
    seeds: tuple[int, ...] = SCREENING_SEEDS,
    heads: tuple[int, ...] = (4, 8),
    epochs: int | None = None,
    patience: int | None = None,
    dry_run: bool = False,
) -> Path:
    """Refine frozen HGT Top-1: heads ∈ {4,8}, everything else frozen.

    heads=4 already exists under shared/ — we still materialize refine/ runs
    for heads=8 (and skip-copy metrics for hd4 from freeze if present).
    """
    base = frozen_hgt_top1()
    jobs: list[tuple[StageAConfig, int]] = []
    for cfg in iter_refine_grid("hgt", base):
        if cfg.heads not in heads:
            continue
        for seed in seeds:
            jobs.append((cfg, seed))

    summary_rows: list[dict[str, Any]] = []
    n_jobs = len(jobs)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for i, (cfg, seed) in enumerate(jobs, 1):
        out_dir = run_dir_for(cfg, scenario, seed)
        # Reuse shared freeze for heads=4 instead of retraining.
        if cfg.heads == 4 and not dry_run:
            shared_cfg = StageAConfig(
                backbone=cfg.backbone,
                hidden_dim=cfg.hidden_dim,
                n_layers=cfg.n_layers,
                dropout=cfg.dropout,
                lr=cfg.lr,
                heads=4,
                phase="shared",
            )
            shared_res = (
                run_dir_for(shared_cfg, scenario, seed) / "result_11.08.2026.json"
            )
            if shared_res.exists():
                prev = json.loads(shared_res.read_text(encoding="utf-8"))
                if prev.get("status") == "ok":
                    out_dir.mkdir(parents=True, exist_ok=True)
                    # Pointer result — metrics from shared freeze, no retrain.
                    pointer = {
                        **prev,
                        "status": "ok",
                        "reused_from_shared": str(shared_res),
                        "phase": "refine",
                        "heads": 4,
                        "note": "Reused Stage A shared freeze (heads=4); not retrained.",
                    }
                    (out_dir / "result_11.08.2026.json").write_text(
                        json.dumps(pointer, indent=2), encoding="utf-8"
                    )
                    print(
                        f"[{i}/{n_jobs}] REUSE shared hd4 {cfg.config_id} seed={seed}"
                    )
                    row = {
                        "config_id": cfg.config_id,
                        **cfg.to_dict(),
                        "scenario": scenario,
                        "training_seed": seed,
                        "status": "ok",
                        "reused_from_shared": True,
                    }
                    if "selection" in prev:
                        row.update(prev["selection"])
                        row["candidate_fingerprint"] = prev.get(
                            "candidate_fingerprint"
                        )
                    summary_rows.append(row)
                    if i % PROGRESS_EVERY == 0 or i == n_jobs:
                        _write_refine_summary(summary_rows, scenario=scenario)
                    continue

        prior = out_dir / "result_11.08.2026.json"
        if prior.exists() and not dry_run:
            try:
                prev = json.loads(prior.read_text(encoding="utf-8"))
            except Exception:
                prev = {}
            if prev.get("status") == "ok":
                print(f"[{i}/{n_jobs}] SKIP ok {cfg.config_id} seed={seed}")
                row = {
                    "config_id": cfg.config_id,
                    **cfg.to_dict(),
                    "scenario": scenario,
                    "training_seed": seed,
                    "status": "ok",
                    "skipped_existing": True,
                }
                if "selection" in prev:
                    row.update(prev["selection"])
                summary_rows.append(row)
                continue

        print(f"[{i}/{n_jobs}] {cfg.config_id} seed={seed} scenario={scenario}")
        res = launch_one(
            cfg,
            scenario=scenario,
            training_seed=seed,
            epochs=epochs,
            patience=patience,
            dry_run=dry_run,
        )
        row = {
            "config_id": cfg.config_id,
            **cfg.to_dict(),
            "scenario": scenario,
            "training_seed": seed,
            "status": res.get("status"),
        }
        if "selection" in res:
            row.update(res["selection"])
            row["candidate_fingerprint"] = res.get("candidate_fingerprint")
        summary_rows.append(row)
        if i % PROGRESS_EVERY == 0:
            _write_refine_summary(summary_rows, scenario=scenario)

    path = _write_refine_summary(summary_rows, scenario=scenario, final=True)
    _write_heads_comparison(summary_rows)
    print(f"Wrote {path}")
    return path


def _write_refine_summary(
    summary_rows: list[dict[str, Any]],
    *,
    scenario: str,
    final: bool = False,
) -> Path:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUT_ROOT / "STAGE_A_HGT_HEADS_REFINE_SUMMARY_11.08.2026.json"
    path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    status = {
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "phase": "stage_a_hgt_heads_refine",
        "scenario": scenario,
        "done": len(summary_rows),
        "ok": sum(1 for r in summary_rows if r.get("status") == "ok"),
        "fail": sum(1 for r in summary_rows if r.get("status") not in {"ok", None}),
        "final": final,
        "wandb_group": "TaskA_FINAL_LARGE_GRID_11_08_2026_STAGE_A",
    }
    (LOG_ROOT).mkdir(parents=True, exist_ok=True)
    (LOG_ROOT / "STAGE_A_HGT_HEADS_REFINE_STATUS_11.08.2026.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    return path


def _write_heads_comparison(summary_rows: list[dict[str, Any]]) -> Path:
    import statistics
    from collections import defaultdict

    by_heads: dict[int, list[float]] = defaultdict(list)
    for r in summary_rows:
        if r.get("status") != "ok" or r.get("valid_auprc") is None:
            continue
        by_heads[int(r["heads"])].append(float(r["valid_auprc"]))

    lines = [
        "# HGT heads refine — 4 vs 8 (11.08.2026)",
        "",
        "Frozen base: `hgt h32 L2 d0.25 lr1e-3` on `clean` (no stats / g_stat=zeros).",
        "heads=4 reused from Stage A shared freeze; heads=8 trained in refine/.",
        "",
        "| heads | n_seeds | mean valid AUPRC ± std |",
        "| ---: | ---: | ---: |",
    ]
    ranking = []
    for hd in sorted(by_heads):
        xs = by_heads[hd]
        m = statistics.mean(xs)
        s = statistics.stdev(xs) if len(xs) > 1 else float("nan")
        ranking.append((m, hd, len(xs), s))
        s_txt = f"{s:.4f}" if s == s else "—"
        lines.append(f"| {hd} | {len(xs)} | {m:.4f} ± {s_txt} |")
    ranking.sort(reverse=True)
    lines += ["", "## Per-seed", ""]
    for r in sorted(
        summary_rows, key=lambda z: (int(z.get("heads") or 0), int(z.get("training_seed") or 0))
    ):
        if r.get("status") != "ok":
            continue
        lines.append(
            f"- hd{r['heads']} seed{r['training_seed']}: "
            f"valid AUPRC={float(r['valid_auprc']):.4f}"
            + (" (reused shared)" if r.get("reused_from_shared") else "")
        )
    if ranking:
        lines += [
            "",
            f"**Leader:** heads={ranking[0][1]} "
            f"(mean valid AUPRC {ranking[0][0]:.4f})",
            "",
            "Note: this is heads refine only — Stage C statistical S0–S10 not started.",
            "",
        ]
    path = OUT_ROOT / "STAGE_A_HGT_HEADS_4vs8_11.08.2026.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Stage A backbone race 11.08.2026")
    p.add_argument(
        "--mode",
        choices=["smoke", "shared_screen", "count", "refine_hgt_heads"],
        default="smoke",
    )
    p.add_argument("--scenario", default="clean")
    p.add_argument("--max-jobs", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--backbone",
        action="append",
        default=None,
        help="Restrict to backbone(s); repeatable",
    )
    args = p.parse_args(argv)

    if args.mode == "count":
        n = count_shared_jobs(3)
        print(f"shared configs × 3 seeds = {n}")
        print(f"unique shared configs = {sum(1 for _ in iter_shared_grid())}")
        return 0

    if args.mode == "smoke":
        # One cheap job: HGT h32 L2 drop0.1 lr1e-3, 2 epochs
        cfg = StageAConfig(
            backbone="hgt",
            hidden_dim=32,
            n_layers=2,
            dropout=0.1,
            lr=1e-3,
            heads=4,
            phase="shared",
        )
        res = launch_one(
            cfg,
            scenario=args.scenario,
            training_seed=SCREENING_SEEDS[0],
            epochs=args.epochs or 2,
            patience=args.patience or 2,
            dry_run=args.dry_run,
        )
        print(json.dumps(res, indent=2))
        return 0 if res.get("status") in {"ok", "dry_run"} else 1

    if args.mode == "refine_hgt_heads":
        run_hgt_heads_refine(
            scenario=args.scenario,
            epochs=args.epochs,
            patience=args.patience,
            dry_run=args.dry_run,
        )
        return 0

    run_shared_screen(
        scenario=args.scenario,
        max_jobs=args.max_jobs,
        epochs=args.epochs,
        patience=args.patience,
        dry_run=args.dry_run,
        backbones=tuple(args.backbone) if args.backbone else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
