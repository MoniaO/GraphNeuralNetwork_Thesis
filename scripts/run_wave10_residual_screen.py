#!/usr/bin/env python3
"""Wave 10B — R0 MLP control vs R1 AG KAN residual (clean, seed 20260722).

Order: Stage 0 diagnostics → R0 → R1.
Promote to 10C (3 seeds) only if Δ(R1−R0) ≥ +0.005, or in (−0.005,+0.005)
with a clear diagnostic justification (manual flag --force-promote).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
OUT = ROOT / "outputs" / "wave10"
DEFAULT_GSN = Path.home() / "Desktop" / "GSN Graphs dysertation 2026"
SEED = 20260722
BASELINE_VALID = 0.9215931791903705
TOL = 0.002
PROMOTE_DELTA = 0.005

R0 = "W10_R0_FROZEN_MLP_CONTROL"
R1 = "W10_R1_AG_KAN_RESIDUAL_FROZEN_MLP"


def _env() -> dict:
    env = os.environ.copy()
    env.setdefault("GSN_PROJECT_ROOT", str(DEFAULT_GSN))
    env.setdefault(
        "PHARMA_DATA_ROOT",
        str(Path(env["GSN_PROJECT_ROOT"]) / "2 v3. Data" / "dataset_v3"),
    )
    env.setdefault("PYTHONPATH", f"{ROOT / 'src'}:{ROOT}")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _metrics_from_ckpt(ckpt: Path) -> dict:
    import torch

    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    fv = blob.get("final_valid_metrics") or {}
    ft = blob.get("final_test_metrics") or {}
    n_params = sum(int(v.numel()) for v in blob["model_state_dict"].values())
    n_train = sum(
        int(v.numel())
        for k, v in blob["model_state_dict"].items()
        # approximate: residual branch keys
        if "ag_kan_residual" in k
    )
    return {
        "best_valid_auprc": float(blob.get("best_valid_metric", float("nan"))),
        "valid_auprc": float(fv.get("auprc", blob.get("best_valid_metric", float("nan")))),
        "valid_brier": float(fv.get("brier", float("nan"))),
        "valid_auroc": float(fv.get("auc", fv.get("auroc", float("nan")))),
        "test_auprc": float(ft.get("auprc", float("nan"))),
        "test_brier": float(ft.get("brier", float("nan"))),
        "parameter_count": n_params,
        "ag_kan_parameter_count": n_train,
        "best_epoch": int(blob.get("best_epoch", -1)),
    }


def run_one(variant: str, *, residual: bool, epochs: int, lr: float | None = None) -> dict:
    run_dir = OUT / "runs" / variant / "clean" / f"seed_{SEED}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    ckpt = run_dir / "best_model.pt"
    if metrics_path.exists() and ckpt.exists():
        print(f"SKIP {variant}", flush=True)
        return json.loads(metrics_path.read_text())

    t0 = time.time()
    cmd = [
        PY,
        str(ROOT / "src" / "train_taskA.py"),
        "model=TaskA_hgt_wave7c",
        "hcr=w7c_b2_audit",
        "data.dataset.scenario=clean",
        "data.feature_ablation_profile=empirical",
        f"data.candidate_seed={SEED}",
        f"training.seed={SEED}",
        "training.device=cpu",
        f"training.epochs={epochs}",
        "training.early_stopping_patience=40",
        "training.grad_clip=1.0",
        "experiment.wave=WAVE10",
        f"experiment.variant={variant}",
        f"experiment.intervention={variant}",
        "experiment.motif_completion.enabled=false",
        "model.decoder.arch=unshared_mlp",
        "model.decoder.ablation=none",
        "model.decoder.use_triple=false",
        "model.decoder.pair_encoder.type=mlp",
        "model.decoder.pair_encoder.input_dim=40",
        "model.decoder.pair_encoder.hidden_dim=16",
        "model.decoder.pair_encoder.output_dim=8",
        "model.decoder.pair_encoder.dropout=0.1",
        f"model.decoder.ag_kan_residual.enabled={'true' if residual else 'false'}",
        "wandb.enabled=true",
        "wandb.group=TaskA_WAVE10_KAN_RESIDUAL",
        "wandb.job_type=wave10_screen",
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
    if residual:
        # R1 starts from the paired R0 MLP checkpoint (frozen backbone).
        r0_ckpt = OUT / "runs" / R0 / "clean" / f"seed_{SEED}" / "best_model.pt"
        if not r0_ckpt.is_file():
            raise SystemExit(f"R1 requires R0 checkpoint at {r0_ckpt}")
        cmd += [
            "model.decoder.ag_kan_residual.input_dim=40",
            "model.decoder.ag_kan_residual.output_dim=8",
            "model.decoder.ag_kan_residual.spline_order=3",
            "model.decoder.ag_kan_residual.grid_size=3",
            "model.decoder.ag_kan_residual.grid_range=[-3.0,3.0]",
            "model.decoder.ag_kan_residual.grid_update=false",
            "model.decoder.ag_kan_residual.base_activation=silu",
            "model.decoder.ag_kan_residual.base_scale_init=0.1",
            "model.decoder.ag_kan_residual.spline_scale_init=0.05",
            "model.decoder.ag_kan_residual.use_bias=true",
            "model.decoder.ag_kan_residual.gate_init_logit=-3.0",
            "model.decoder.ag_kan_residual.spline_l1=1.0e-5",
            "model.decoder.ag_kan_residual.type_routed=false",
            "model.decoder.ag_kan_residual.train_mode=residual_only",
            f"training.init_from_checkpoint={r0_ckpt}",
        ]
    if lr is not None:
        cmd.append(f"training.lr={lr}")

    print("\nRUN:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), env=_env())
    runtime_s = time.time() - t0
    if proc.returncode != 0:
        raise SystemExit(f"Failed {variant} rc={proc.returncode}")
    row = {
        "wave": "WAVE10",
        "variant": variant,
        "ag_kan_residual": residual,
        "scenario": "clean",
        "seed": SEED,
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--skip-diagnose", action="store_true")
    ap.add_argument("--force-r1", action="store_true")
    ap.add_argument("--force-promote", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summaries").mkdir(exist_ok=True)
    (OUT / "audits").mkdir(exist_ok=True)

    if not args.skip_diagnose:
        print("Wave 10A — diagnosing Wave 9 K1…", flush=True)
        proc = subprocess.run(
            [PY, str(ROOT / "scripts" / "diagnose_wave10_kan_failure.py")],
            cwd=str(ROOT),
            env=_env(),
        )
        if proc.returncode != 0:
            raise SystemExit(f"diagnose failed rc={proc.returncode}")

    csv_path = OUT / "summaries" / "wave10_screen.csv"
    r0 = run_one(R0, residual=False, epochs=args.epochs)
    append_csv(csv_path, r0)
    delta_repro = abs(float(r0["valid_auprc"]) - BASELINE_VALID)
    repro = {
        "baseline_valid_auprc": BASELINE_VALID,
        "r0_valid_auprc": r0["valid_auprc"],
        "abs_delta": delta_repro,
        "tolerance": TOL,
        "passed": bool(delta_repro < TOL),
    }
    (OUT / "audits" / "r0_reproduction.json").write_text(json.dumps(repro, indent=2))
    print(json.dumps(repro, indent=2), flush=True)
    if not repro["passed"] and not args.force_r1:
        raise SystemExit(f"R0 failed reproduction gate |Δ|={delta_repro:.6f} >= {TOL}")

    r1 = run_one(R1, residual=True, epochs=args.epochs)
    append_csv(csv_path, r1)

    d = float(r1["valid_auprc"]) - float(r0["valid_auprc"])
    if d >= PROMOTE_DELTA:
        action = "promote_multiseed"
    elif d < -PROMOTE_DELTA:
        action = "stop"
    else:
        action = "diagnostic_only" if not args.force_promote else "promote_multiseed"

    decision = {
        "selection_metric": "valid_auprc",
        "R0_valid": r0["valid_auprc"],
        "R1_valid": r1["valid_auprc"],
        "R1_minus_R0": d,
        "R0_test": r0.get("test_auprc"),
        "R1_test": r1.get("test_auprc"),
        "R0_brier": r0.get("valid_brier"),
        "R1_brier": r1.get("valid_brier"),
        "R0_params": r0.get("parameter_count"),
        "R1_params": r1.get("parameter_count"),
        "ag_kan_params": r1.get("ag_kan_parameter_count"),
        "action": action,
        "promote_to_multiseed": action == "promote_multiseed",
        "thresholds": {"promote_delta": PROMOTE_DELTA, "stop_below": -PROMOTE_DELTA},
        "note": (
            "Δ>=+0.005 → 10C three seeds; Δ<-0.005 → stop; "
            "else only with diagnostic justification."
        ),
    }
    (OUT / "summaries" / "WAVE10_DECISION.json").write_text(json.dumps(decision, indent=2))
    (OUT / "WAVE10_DECISION.json").write_text(json.dumps(decision, indent=2))
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
