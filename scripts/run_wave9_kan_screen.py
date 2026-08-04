#!/usr/bin/env python3
"""Wave 9A — K0 MLP control vs K1 fixed-grid pair KAN (clean, seed 20260722).

Gate: |valid_AUPRC(K0) - 0.921593| < 0.002 before launching K1.
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
OUT = ROOT / "outputs" / "wave9"
DEFAULT_GSN = Path.home() / "Desktop" / "GSN Graphs dysertation 2026"
SEED = 20260722
BASELINE_VALID = 0.9215931791903705
TOL = 0.002

K0 = ("W9_K0_MLP_CONTROL", "mlp")
K1 = ("W9_K1_PAIR_KAN_FIXED", "kan")


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
    return {
        "best_valid_auprc": float(blob.get("best_valid_metric", float("nan"))),
        "valid_auprc": float(fv.get("auprc", blob.get("best_valid_metric", float("nan")))),
        "valid_brier": float(fv.get("brier", float("nan"))),
        "valid_auroc": float(fv.get("auc", fv.get("auroc", float("nan")))),
        "test_auprc": float(ft.get("auprc", float("nan"))),
        "test_brier": float(ft.get("brier", float("nan"))),
        "parameter_count": n_params,
    }


def run_one(variant: str, pe_type: str, epochs: int, lr: float | None = None) -> dict:
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
        "experiment.wave=WAVE9",
        f"experiment.variant={variant}",
        f"experiment.intervention={variant}",
        "experiment.motif_completion.enabled=false",
        "model.decoder.arch=unshared_mlp",
        "model.decoder.ablation=none",
        "model.decoder.use_triple=false",
        f"model.decoder.pair_encoder.type={pe_type}",
        "model.decoder.pair_encoder.input_dim=40",
        "model.decoder.pair_encoder.hidden_dim=16",
        "model.decoder.pair_encoder.output_dim=8",
        "model.decoder.pair_encoder.dropout=0.1",
        "wandb.enabled=true",
        "wandb.group=TaskA_WAVE9_KAN_SCREEN",
        "wandb.job_type=wave9_screen",
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
    if pe_type == "kan":
        cmd += [
            "model.decoder.pair_encoder.spline_order=3",
            "model.decoder.pair_encoder.grid_size=5",
            "model.decoder.pair_encoder.grid_range=[-3.0,3.0]",
            "model.decoder.pair_encoder.grid_update=false",
            "model.decoder.pair_encoder.base_activation=silu",
            "model.decoder.pair_encoder.base_scale_init=1.0",
            "model.decoder.pair_encoder.spline_scale_init=0.1",
            "model.decoder.pair_encoder.use_bias=true",
            "model.decoder.pair_encoder.spline_l1=1.0e-5",
        ]
    if lr is not None:
        cmd.append(f"training.lr={lr}")

    print("\nRUN:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), env=_env())
    runtime_s = time.time() - t0
    if proc.returncode != 0:
        raise SystemExit(f"Failed {variant} rc={proc.returncode}")
    row = {
        "wave": "WAVE9",
        "variant": variant,
        "pair_encoder": pe_type,
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
    ap.add_argument("--skip-audit", action="store_true")
    ap.add_argument("--force-k1", action="store_true", help="Run K1 even if K0 gate fails")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summaries").mkdir(exist_ok=True)
    (OUT / "audits").mkdir(exist_ok=True)

    if not args.skip_audit:
        print("Running input range audit…", flush=True)
        proc = subprocess.run(
            [PY, str(ROOT / "scripts" / "audit_wave9_input_ranges.py")],
            cwd=str(ROOT),
            env=_env(),
        )
        if proc.returncode != 0:
            raise SystemExit(f"audit failed rc={proc.returncode}")

    csv_path = OUT / "summaries" / "wave9_single_seed.csv"
    k0 = run_one(K0[0], K0[1], args.epochs)
    append_csv(csv_path, k0)
    delta = abs(float(k0["valid_auprc"]) - BASELINE_VALID)
    repro = {
        "baseline_valid_auprc": BASELINE_VALID,
        "k0_valid_auprc": k0["valid_auprc"],
        "abs_delta": delta,
        "tolerance": TOL,
        "passed": bool(delta < TOL),
        "k0_test_auprc": k0.get("test_auprc"),
        "k0_parameter_count": k0.get("parameter_count"),
    }
    (OUT / "audits" / "baseline_reproduction.json").write_text(json.dumps(repro, indent=2))
    print(json.dumps(repro, indent=2), flush=True)

    if not repro["passed"] and not args.force_k1:
        raise SystemExit(
            f"K0 failed reproduction gate |Δ|={delta:.6f} >= {TOL}. "
            "Fix config before KAN."
        )

    k1 = run_one(K1[0], K1[1], args.epochs)
    append_csv(csv_path, k1)

    decision = {
        "selection_metric": "valid_auprc",
        "K0_valid": k0["valid_auprc"],
        "K1_valid": k1["valid_auprc"],
        "K1_minus_K0": float(k1["valid_auprc"]) - float(k0["valid_auprc"]),
        "K0_test": k0.get("test_auprc"),
        "K1_test": k1.get("test_auprc"),
        "K0_params": k0.get("parameter_count"),
        "K1_params": k1.get("parameter_count"),
        "promote_to_multiseed": bool(
            float(k1["valid_auprc"]) + 1e-12 >= float(k0["valid_auprc"]) - 0.01
        ),
        "note": "Promote to 9B if K1 stable and not clearly inferior (≤1pp).",
    }
    (OUT / "summaries" / "WAVE9A_DECISION.json").write_text(json.dumps(decision, indent=2))
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
