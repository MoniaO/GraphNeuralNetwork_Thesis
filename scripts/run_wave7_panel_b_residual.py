#!/usr/bin/env python3
"""Wave 7 Panel B residual fusion — clean audit (R1 vs R2).

Reference (do not retrain):
  B0 valid AUPRC = 0.804
  B2 valid AUPRC = 0.791

Protocol
--------
- seed 20260722, scenario clean first
- W&B group TaskA_WAVE7_PANEL_B_RESIDUAL
- select on validation AUPRC only
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
    ("W7BR_R1_B2_SHARED_PAIR_ENCODER", "w7br_r1_b2_shared_pair_encoder", False),
    ("W7BR_R2_B2_GATED_LEGACY_RESIDUAL", "w7br_r2_b2_gated_legacy_residual", True),
]

REF = {
    "B0_valid_auprc": 0.8043239824945645,
    "B2_valid_auprc": 0.7905495945616131,
}


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


def run_one(variant: str, hcr_name: str, use_legacy: bool, scenario: str, epochs: int) -> dict:
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
        f"model.decoder.use_legacy_residual={str(use_legacy).lower()}",
        "wandb.enabled=true",
        "wandb.group=TaskA_WAVE7_PANEL_B_RESIDUAL",
        "wandb.job_type=wave7_panel_b_residual",
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
        raise SystemExit(f"Failed {variant} {scenario} rc={proc.returncode}")
    if not ckpt.exists():
        raise SystemExit(f"Missing checkpoint {ckpt}")
    row = {
        "wave": "WAVE7_PANEL_B_RESIDUAL",
        "variant": variant,
        "scenario": scenario,
        "seed": SEED,
        "arch_tag": "joint_L3_H8_W32_leaky_relu",
        "use_legacy_residual": use_legacy,
        **_metrics_from_ckpt(ckpt),
        **REF,
    }
    metrics_path.write_text(json.dumps(row, indent=2))
    return row


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
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
    (OUT / "WAVE7_PANEL_B_RESIDUAL_PROTOCOL.json").write_text(
        json.dumps(
            {
                "wave": "WAVE7_PANEL_B_RESIDUAL",
                "wandb_group": "TaskA_WAVE7_PANEL_B_RESIDUAL",
                "seed": SEED,
                "scenarios": scenarios,
                "variants": [v[0] for v in VARIANTS],
                "reference": REF,
                "selection": "validation AUPRC; compare R2−R1 for legacy residual gain",
                "panel_b_decision_unchanged": "B0 remains formal Panel B winner",
            },
            indent=2,
        )
    )

    csv_path = OUT / "wave7_panel_b_residual_results.csv"
    rows = []
    for scenario in scenarios:
        for variant, hcr_name, use_legacy in VARIANTS:
            row = run_one(variant, hcr_name, use_legacy, scenario, args.epochs)
            append_csv(csv_path, row)
            rows.append(row)
            print(
                f"DONE {variant}: valid={row.get('valid_auprc')} test={row.get('test_auprc')}",
                flush=True,
            )

    clean = [r for r in rows if r.get("scenario") == "clean"]
    if len(clean) >= 2:
        by_v = {r["variant"]: r for r in clean}
        r1 = by_v.get("W7BR_R1_B2_SHARED_PAIR_ENCODER", {})
        r2 = by_v.get("W7BR_R2_B2_GATED_LEGACY_RESIDUAL", {})
        decision = {
            "selection_metric": "valid_auprc",
            "R1_valid_auprc": r1.get("valid_auprc"),
            "R2_valid_auprc": r2.get("valid_auprc"),
            "delta_R2_minus_R1": (
                float(r2["valid_auprc"]) - float(r1["valid_auprc"])
                if r1 and r2
                else None
            ),
            "reference_B0_valid": REF["B0_valid_auprc"],
            "reference_B2_valid": REF["B2_valid_auprc"],
            "panel_b_formal_winner_unchanged": "W7B_B0_LEGACY_V0_PAD40",
            "rows": clean,
        }
        # Interpret cases
        r1v = float(r1.get("valid_auprc", float("nan")))
        r2v = float(r2.get("valid_auprc", float("nan")))
        b2v = REF["B2_valid_auprc"]
        b0v = REF["B0_valid_auprc"]
        notes = []
        if r1v > b2v + 1e-4:
            notes.append("Case1: R1>B2 — shared pair encoder helps")
        if r2v > r1v + 1e-4:
            notes.append("Case2: R2>R1 — gated legacy residual helps")
        elif abs(r2v - r1v) <= 1e-4:
            notes.append("Case3: R2≈R1 — V0 residual redundant or gated off")
        else:
            notes.append("Case4: R2<R1 — legacy residual hurts")
        if r2v >= b0v - 1e-4:
            notes.append("Ideal-ish: R2 overall ≥ B0 on valid")
        decision["interpretation"] = notes
        decision["selected_for_report"] = (
            "W7BR_R2_B2_GATED_LEGACY_RESIDUAL"
            if r2v >= r1v
            else "W7BR_R1_B2_SHARED_PAIR_ENCODER"
        )
        (OUT / "WAVE7_PANEL_B_RESIDUAL_DECISION.json").write_text(
            json.dumps(decision, indent=2)
        )
        print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
