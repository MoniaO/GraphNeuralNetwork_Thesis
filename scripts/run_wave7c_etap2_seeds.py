#!/usr/bin/env python3
"""Wave 7C Etap 2 — freeze context-safe architecture, 3 seeds on clean.

Frozen models (from Etap 1 screens):
  T0  W7C_T0_SHARED_MLP_ROLE_MASKS
      = A0 shared 40→16→8 + C6 explicit role masks
      Passed: C3≪C0, C5≪C0; C6≈C0 (bias not required for score).
  A1  W7C_A1_UNSHARED_ROLE_ENCODERS
      Best architecture screen (valid 0.922 on seed 20260722).

Selection remains validation AUPRC only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
OUT = ROOT / "outputs" / "wave7" / "wave7c" / "etap2"
SEEDS = (20260722, 20260723, 20260724)

FROZEN = [
    # Context-safe production freeze (shared MLP + exact-zero missing roles).
    ("W7C_T0_SHARED_MLP_ROLE_MASKS", "shared_mlp", "explicit_role_masks"),
    # Architecture winner from A0–A3 (nearly tied with A0).
    ("W7C_A1_UNSHARED_ROLE_ENCODERS", "unshared_mlp", "none"),
]

# Seed-22 aliases already trained in Etap 1 under different names.
SEED22_ALIASES = {
    ("W7C_T0_SHARED_MLP_ROLE_MASKS", 20260722): (
        ROOT
        / "outputs/wave7/wave7c/runs/W7C_C6_EXPLICIT_ROLE_MASKS/clean/seed_20260722"
    ),
    ("W7C_A1_UNSHARED_ROLE_ENCODERS", 20260722): (
        ROOT
        / "outputs/wave7/wave7c/runs/W7C_A1_UNSHARED_ROLE_ENCODERS/clean/seed_20260722"
    ),
}


def _metrics_from_ckpt(ckpt: Path) -> dict:
    import torch

    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    fv = blob.get("final_valid_metrics") or {}
    ft = blob.get("final_test_metrics") or {}
    n_params = sum(int(v.numel()) for v in blob["model_state_dict"].values())
    return {
        "best_valid_auprc": float(blob.get("best_valid_metric", float("nan"))),
        "valid_auprc": float(fv.get("auprc", blob.get("best_valid_metric", float("nan")))),
        "valid_brier": float(fv.get("brier", float("nan"))),
        "valid_auroc": float(fv.get("auc", fv.get("auroc", float("nan")))),
        "test_auprc": float(ft.get("auprc", float("nan"))),
        "test_brier": float(ft.get("brier", float("nan"))),
        "parameter_count": n_params,
    }


def run_one(variant: str, arch: str, ablation: str, seed: int, epochs: int) -> dict:
    run_dir = OUT / "runs" / variant / "clean" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    ckpt = run_dir / "best_model.pt"

    if metrics_path.exists() and ckpt.exists():
        print(f"SKIP {variant} seed={seed}", flush=True)
        return json.loads(metrics_path.read_text())

    # Reuse Etap-1 seed-22 checkpoints when identical.
    alias = SEED22_ALIASES.get((variant, seed))
    if alias is not None and (alias / "best_model.pt").exists():
        print(f"REUSE etap1 {alias} → {run_dir}", flush=True)
        import shutil

        for name in ("best_model.pt", "metrics.json"):
            src = alias / name
            if src.exists():
                shutil.copy2(src, run_dir / name)
        # Rewrite metrics with etap2 variant/seed tags.
        row = {
            "wave": "WAVE7C_ETAP2",
            "variant": variant,
            "arch": arch,
            "ablation": ablation,
            "scenario": "clean",
            "seed": seed,
            "reused_from": str(alias.relative_to(ROOT)),
            **_metrics_from_ckpt(run_dir / "best_model.pt"),
        }
        metrics_path.write_text(json.dumps(row, indent=2))
        return row

    t0 = time.time()
    cmd = [
        PY,
        str(ROOT / "src" / "train_taskA.py"),
        "model=TaskA_hgt_wave7c",
        "hcr=w7c_b2_audit",
        "data.dataset.scenario=clean",
        "data.feature_ablation_profile=empirical",
        f"data.candidate_seed={seed}",
        f"training.seed={seed}",
        "training.device=cpu",
        f"training.epochs={epochs}",
        "training.early_stopping_patience=40",
        "training.grad_clip=1.0",
        "experiment.wave=WAVE7C",
        f"experiment.variant={variant}",
        f"experiment.intervention={variant}",
        "experiment.motif_completion.enabled=false",
        f"model.decoder.arch={arch}",
        f"model.decoder.ablation={ablation}",
        "wandb.enabled=true",
        "wandb.group=TaskA_WAVE7C_ETAP2_SEEDS",
        "wandb.job_type=wave7c_etap2",
        f"hydra.run.dir={run_dir}",
        "model.hgt.hidden_dim=32",
        "model.hgt.num_layers=3",
        "model.hgt.heads=8",
        "model.hgt.activation=leaky_relu",
        "model.hgt.dropout=0.2",
        "model.hgt.residual=true",
        "model.num_layers=3",
        "model.hidden_dim=32",
        "model.hidden_channels=32",
        "model.heads=8",
    ]
    print("\nRUN:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT))
    runtime_s = time.time() - t0
    if proc.returncode != 0:
        raise SystemExit(f"Failed {variant} seed={seed} rc={proc.returncode}")
    row = {
        "wave": "WAVE7C_ETAP2",
        "variant": variant,
        "arch": arch,
        "ablation": ablation,
        "scenario": "clean",
        "seed": seed,
        "runtime_s": runtime_s,
        **_metrics_from_ckpt(ckpt),
    }
    metrics_path.write_text(json.dumps(row, indent=2))
    return row


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def _mean_sd(vals: list[float]) -> tuple[float, float]:
    vals = [float(v) for v in vals if v == v]
    if not vals:
        return float("nan"), float("nan")
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return m, math.sqrt(var)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument(
        "--variants",
        default="all",
        help="Comma-separated variant names or 'all'.",
    )
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    freeze_note = {
        "etap": 2,
        "selection_metric": "valid_auprc",
        "scenario": "clean",
        "seeds": list(SEEDS),
        "frozen": [
            {
                "variant": "W7C_T0_SHARED_MLP_ROLE_MASKS",
                "why": (
                    "Context-safe A0: shared nonlinear pair encoder + explicit "
                    "role masks so missing AZ/ZG encode as exact zero latents. "
                    "Etap1: C3≪C0 (not type/support shortcut), C5≪C0 (uses "
                    "dependence), C4≈C0 (specific Z weak), C6≈C0."
                ),
            },
            {
                "variant": "W7C_A1_UNSHARED_ROLE_ENCODERS",
                "why": (
                    "Architecture screen winner on seed 20260722 "
                    "(valid 0.922 vs A0 0.918). Confirm across seeds."
                ),
            },
        ],
        "etap1_refs": {
            "C0_valid": 0.919,
            "C3_valid": 0.675,
            "C5_valid": 0.665,
            "C6_valid": 0.911,
            "A1_valid": 0.922,
        },
    }
    (OUT / "ETAP2_FREEZE.json").write_text(json.dumps(freeze_note, indent=2))

    variants = FROZEN
    if args.variants != "all":
        wanted = {v.strip() for v in args.variants.split(",") if v.strip()}
        variants = [v for v in FROZEN if v[0] in wanted]

    csv_path = OUT / "wave7c_etap2_seeds.csv"
    rows = []
    for variant, arch, ablation in variants:
        for seed in SEEDS:
            row = run_one(variant, arch, ablation, seed, args.epochs)
            append_csv(csv_path, row)
            rows.append(row)
            print(
                f"DONE {variant} seed={seed}: valid={row.get('valid_auprc')}",
                flush=True,
            )

    summary: dict = {"selection_metric": "valid_auprc", "by_variant": {}}
    for variant, _, _ in variants:
        sub = [r for r in rows if r["variant"] == variant]
        va = [r["valid_auprc"] for r in sub]
        ta = [r["test_auprc"] for r in sub]
        vm, vs = _mean_sd(va)
        tm, ts = _mean_sd(ta)
        summary["by_variant"][variant] = {
            "n_seeds": len(sub),
            "valid_auprc_mean": vm,
            "valid_auprc_sd": vs,
            "test_auprc_mean": tm,
            "test_auprc_sd": ts,
            "seeds": {str(r["seed"]): r["valid_auprc"] for r in sub},
        }
    # Pick T0 by mean valid AUPRC among frozen.
    best_v, best = max(
        summary["by_variant"].items(),
        key=lambda kv: kv[1]["valid_auprc_mean"],
    )
    summary["selected_for_etap3"] = best_v
    summary["selected_valid_mean_sd"] = {
        "mean": best["valid_auprc_mean"],
        "sd": best["valid_auprc_sd"],
    }
    (OUT / "ETAP2_SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
