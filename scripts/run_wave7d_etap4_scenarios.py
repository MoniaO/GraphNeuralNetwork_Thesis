#!/usr/bin/env python3
"""Wave 7D Etap 4 — frozen A1 on all patient scenarios × 3 seeds.

Winner (validation selection):
  W7D_T0_A1_UNSHARED_ROLE_ENCODERS  (unshared 40→16→8 role encoders)

No CMI / residual / KAN here — architecture and features stay frozen.
Report mean ± SD of valid/test AUPRC per scenario.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
OUT = ROOT / "outputs" / "wave7" / "wave7c" / "etap4"
DEFAULT_GSN = Path.home() / "Desktop" / "GSN Graphs dysertation 2026"

VARIANT = "W7D_T0_A1_UNSHARED_ROLE_ENCODERS"
ARCH = "unshared_mlp"
ABLATION = "none"
USE_TRIPLE = False

SEEDS = (20260722, 20260723, 20260724)
SCENARIOS = (
    "clean",
    "hidden_confounder",
    "selection_bias",
    "no_overlap",
    "noisy_documentation",
    "multihospital",
)

# Reuse Etap-2 clean A1 checkpoints when present.
ETAP2_A1 = ROOT / "outputs/wave7/wave7c/etap2/runs/W7C_A1_UNSHARED_ROLE_ENCODERS"


def _train_env() -> dict:
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


def run_one(scenario: str, seed: int, epochs: int) -> dict:
    run_dir = OUT / "runs" / VARIANT / scenario / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    ckpt = run_dir / "best_model.pt"
    if metrics_path.exists() and ckpt.exists():
        print(f"SKIP {VARIANT} {scenario} seed={seed}", flush=True)
        return json.loads(metrics_path.read_text())

    alias = ETAP2_A1 / scenario / f"seed_{seed}"
    if scenario == "clean" and (alias / "best_model.pt").exists():
        print(f"REUSE etap2 {alias} → {run_dir}", flush=True)
        shutil.copy2(alias / "best_model.pt", ckpt)
        row = {
            "wave": "WAVE7D_ETAP4",
            "variant": VARIANT,
            "arch": ARCH,
            "ablation": ABLATION,
            "use_triple": USE_TRIPLE,
            "scenario": scenario,
            "seed": seed,
            "reused_from": str(alias.relative_to(ROOT)),
            **_metrics_from_ckpt(ckpt),
        }
        metrics_path.write_text(json.dumps(row, indent=2))
        return row

    t0 = time.time()
    cmd = [
        PY,
        str(ROOT / "src" / "train_taskA.py"),
        "model=TaskA_hgt_wave7c",
        "hcr=w7c_b2_audit",
        f"data.dataset.scenario={scenario}",
        "data.feature_ablation_profile=empirical",
        f"data.candidate_seed={seed}",
        f"training.seed={seed}",
        "training.device=cpu",
        f"training.epochs={epochs}",
        "training.early_stopping_patience=40",
        "training.grad_clip=1.0",
        "experiment.wave=WAVE7D",
        f"experiment.variant={VARIANT}",
        f"experiment.intervention={VARIANT}",
        "experiment.motif_completion.enabled=false",
        f"model.decoder.arch={ARCH}",
        f"model.decoder.ablation={ABLATION}",
        f"model.decoder.use_triple={str(USE_TRIPLE).lower()}",
        "wandb.enabled=true",
        "wandb.group=TaskA_WAVE7D_ETAP4_SCENARIOS",
        "wandb.job_type=wave7d_etap4",
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
    proc = subprocess.run(cmd, cwd=str(ROOT), env=_train_env())
    runtime_s = time.time() - t0
    if proc.returncode != 0:
        raise SystemExit(f"Failed {scenario} seed={seed} rc={proc.returncode}")
    row = {
        "wave": "WAVE7D_ETAP4",
        "variant": VARIANT,
        "arch": ARCH,
        "ablation": ABLATION,
        "use_triple": USE_TRIPLE,
        "scenario": scenario,
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
        "--scenarios",
        default="all",
        help="Comma-separated scenarios or 'all'.",
    )
    ap.add_argument(
        "--seeds",
        default="all",
        help="Comma-separated seeds or 'all'.",
    )
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    scenarios = list(SCENARIOS) if args.scenarios == "all" else [
        s.strip() for s in args.scenarios.split(",") if s.strip()
    ]
    seeds = list(SEEDS) if args.seeds == "all" else [
        int(s.strip()) for s in args.seeds.split(",") if s.strip()
    ]

    protocol = {
        "etap": 4,
        "variant": VARIANT,
        "arch": ARCH,
        "selection_basis": "Etap2/3 validation AUPRC — A1 over T1 CMI",
        "scenarios": scenarios,
        "seeds": seeds,
        "report": "mean ± SD valid/test AUPRC per scenario",
        "kan": "deferred until after scenario matrix",
    }
    (OUT / "ETAP4_PROTOCOL.json").write_text(json.dumps(protocol, indent=2))

    csv_path = OUT / "wave7d_etap4_scenarios.csv"
    rows = []
    for scenario in scenarios:
        for seed in seeds:
            row = run_one(scenario, seed, args.epochs)
            append_csv(csv_path, row)
            rows.append(row)
            print(
                f"DONE {scenario} seed={seed}: "
                f"valid={row.get('valid_auprc')} test={row.get('test_auprc')}",
                flush=True,
            )

    by_sc: dict = {}
    for scenario in scenarios:
        sub = [r for r in rows if r["scenario"] == scenario]
        vm, vs = _mean_sd([r["valid_auprc"] for r in sub])
        tm, ts = _mean_sd([r["test_auprc"] for r in sub])
        by_sc[scenario] = {
            "n_seeds": len(sub),
            "valid_auprc_mean": vm,
            "valid_auprc_sd": vs,
            "test_auprc_mean": tm,
            "test_auprc_sd": ts,
            "seeds": {
                str(r["seed"]): {
                    "valid_auprc": r["valid_auprc"],
                    "test_auprc": r["test_auprc"],
                }
                for r in sub
            },
        }

    summary = {
        "variant": VARIANT,
        "by_scenario": by_sc,
        "next": "Etap 5 calibration / final edges; Etap 6 KAN swap on pair encoder only",
    }
    (OUT / "ETAP4_SUMMARY.json").write_text(json.dumps(summary, indent=2))

    # Compact markdown table for quick reading.
    lines = [
        f"# Etap 4 — {VARIANT}",
        "",
        "| Scenario | valid AUPRC | test AUPRC |",
        "|---|---:|---:|",
    ]
    for sc in scenarios:
        s = by_sc[sc]
        lines.append(
            f"| {sc} | {s['valid_auprc_mean']:.3f} ± {s['valid_auprc_sd']:.3f} "
            f"| {s['test_auprc_mean']:.3f} ± {s['test_auprc_sd']:.3f} |"
        )
    (OUT / "ETAP4_SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
