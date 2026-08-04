#!/usr/bin/env python3
"""Wave 7 Panel B residual — R3 classical4 vs R4 + conditional edge gain.

Primary comparison: R4 − R3 on validation AUPRC (clean, seed 20260722).
Does not retrain or alter R1/R2.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
OUT = ROOT / "outputs" / "wave7" / "panel_b_residual"
SEED = 20260722

VARIANTS = [
    (
        "W7BR_R3_B2_CLASSICAL4_RESIDUAL",
        "w7br_r3_b2_classical4_residual",
        "classical4",
        False,
    ),
    (
        "W7BR_R4_B2_CLASSICAL4_PLUS_CONDITIONAL_EDGE_GAIN",
        "w7br_r4_b2_classical4_plus_conditional_edge_gain",
        "classical4",
        True,
    ),
]


def _metrics_from_ckpt(ckpt: Path) -> dict:
    import torch

    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    final_test = blob.get("final_test_metrics") or {}
    final_valid = blob.get("final_valid_metrics") or {}
    return {
        "best_valid_auprc": float(blob.get("best_valid_metric", float("nan"))),
        "threshold_valid": float(blob.get("classification_threshold", float("nan"))),
        "valid_auprc": float(final_valid.get("auprc", blob.get("best_valid_metric", float("nan")))),
        "valid_auroc": float(final_valid.get("auc", final_valid.get("auroc", float("nan")))),
        "valid_brier": float(final_valid.get("brier", float("nan"))),
        "test_auprc": float(final_test.get("auprc", float("nan"))),
        "test_auroc": float(final_test.get("auc", final_test.get("auroc", float("nan")))),
        "test_brier": float(final_test.get("brier", float("nan"))),
    }


def run_one(variant, hcr_name, legacy_mode, use_triple, scenario, epochs) -> dict:
    run_dir = OUT / "runs" / variant / scenario / f"seed_{SEED}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    ckpt = run_dir / "best_model.pt"
    if metrics_path.exists() and ckpt.exists():
        print(f"SKIP {variant} {scenario}", flush=True)
        return json.loads(metrics_path.read_text())

    cmd = [
        PY,
        str(ROOT / "src" / "train_taskA.py"),
        "model=TaskA_hgt_wave7b_residual",
        f"hcr={hcr_name}",
        f"data.dataset.scenario={scenario}",
        "data.feature_ablation_profile=empirical",
        f"data.candidate_seed={SEED}",
        f"training.seed={SEED}",
        "training.device=cpu",
        f"training.epochs={epochs}",
        "training.early_stopping_patience=40",
        "training.grad_clip=1.0",
        "training.lr=0.001",
        "training.residual_lr=0.0003",
        "experiment.wave=WAVE7_PANEL_B_RESIDUAL",
        f"experiment.variant={variant}",
        f"experiment.intervention={variant}",
        "experiment.motif_completion.enabled=false",
        "model.decoder.use_legacy_residual=true",
        f"model.decoder.legacy_mode={legacy_mode}",
        f"model.decoder.use_triple={str(use_triple).lower()}",
        "wandb.enabled=true",
        "wandb.group=TaskA_WAVE7_PANEL_B_RESIDUAL",
        "wandb.job_type=wave7_panel_b_residual_r3r4",
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
    if proc.returncode != 0:
        raise SystemExit(f"Failed {variant} rc={proc.returncode}")
    row = {
        "wave": "WAVE7_PANEL_B_RESIDUAL",
        "variant": variant,
        "scenario": scenario,
        "seed": SEED,
        "legacy_mode": legacy_mode,
        "use_triple": use_triple,
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
    ap.add_argument("--scenarios", default="clean")
    args = ap.parse_args()
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "wave7_panel_b_residual_r3r4_results.csv"
    rows = []
    for scenario in scenarios:
        for variant, hcr_name, legacy_mode, use_triple in VARIANTS:
            row = run_one(variant, hcr_name, legacy_mode, use_triple, scenario, args.epochs)
            append_csv(csv_path, row)
            rows.append(row)
            print(
                f"DONE {variant}: valid={row.get('valid_auprc')} test={row.get('test_auprc')}",
                flush=True,
            )
    clean = [r for r in rows if r.get("scenario") == "clean"]
    by_v = {r["variant"]: r for r in clean}
    r3 = by_v.get("W7BR_R3_B2_CLASSICAL4_RESIDUAL", {})
    r4 = by_v.get("W7BR_R4_B2_CLASSICAL4_PLUS_CONDITIONAL_EDGE_GAIN", {})
    decision = {
        "selection_metric": "valid_auprc",
        "primary_comparison": "R4 vs R3",
        "R3_valid_auprc": r3.get("valid_auprc"),
        "R4_valid_auprc": r4.get("valid_auprc"),
        "delta_R4_minus_R3": (
            float(r4["valid_auprc"]) - float(r3["valid_auprc"]) if r3 and r4 else None
        ),
        "panel_b_formal_winner_unchanged": "W7B_B0_LEGACY_V0_PAD40",
        "rows": clean,
    }
    (OUT / "WAVE7_PANEL_B_RESIDUAL_R3R4_DECISION.json").write_text(
        json.dumps(decision, indent=2)
    )
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
