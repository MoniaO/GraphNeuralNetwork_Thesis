#!/usr/bin/env python3
"""Post–Wave 7 control: run V1B matched-aux on clean only."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
OUT = ROOT / "outputs" / "wave7"
SEED = 20260722
VARIANT = "W7_V1B_GHCR_BINARY_MATCHED_AUX"
HCR = "w7_v1b_ghcr_binary_matched_aux"
SCENARIO = "clean"


def main() -> int:
    run_dir = OUT / "runs" / VARIANT / SCENARIO / f"seed_{SEED}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        print("SKIP V1B clean — metrics exist:", metrics_path)
        print(metrics_path.read_text())
        return 0

    cmd = [
        PY,
        str(ROOT / "src" / "train_taskA.py"),
        "model=TaskA_hgt_wave7",
        f"hcr={HCR}",
        f"data.dataset.scenario={SCENARIO}",
        "data.feature_ablation_profile=empirical",
        f"data.candidate_seed={SEED}",
        f"training.seed={SEED}",
        "training.device=cpu",
        "training.epochs=300",
        "training.early_stopping_patience=40",
        "experiment.wave=WAVE7",
        f"experiment.variant={VARIANT}",
        f"experiment.intervention={VARIANT}",
        "experiment.motif_completion.enabled=false",
        "wandb.enabled=true",
        "wandb.group=TaskA_WAVE7_GHCR",
        "wandb.job_type=wave7_v1b_control",
        f"hydra.run.dir={run_dir}",
        "model.hgt.hidden_dim=32",
        "model.hgt.num_layers=3",
        "model.hgt.heads=8",
        "model.hgt.activation=leaky_relu",
        "model.hgt.dropout=0.2",
        "model.hgt.residual=true",
        "model.decoder.hidden_dims=[256,128]",
        "model.decoder.activation=gelu",
        "model.decoder.dropout=0.2",
        "model.num_layers=3",
        "model.hidden_dim=32",
        "model.hidden_channels=32",
        "model.heads=8",
    ]
    print("RUN V1B clean:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        return proc.returncode

    import torch

    ckpt = run_dir / "best_model.pt"
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    final_test = blob.get("final_test_metrics") or {}
    final_valid = blob.get("final_valid_metrics") or {}
    row = {
        "wave": "WAVE7",
        "variant": VARIANT,
        "scenario": SCENARIO,
        "seed": SEED,
        "arch_tag": "joint_L3_H8_W32_leaky_relu",
        "best_valid_auprc": float(blob.get("best_valid_metric", float("nan"))),
        "valid_auprc": float(final_valid.get("auprc", blob.get("best_valid_metric", float("nan")))),
        "test_auprc": float(final_test.get("auprc", float("nan"))),
        "test_auroc": float(final_test.get("auc", float("nan"))),
        "note": "control: GHCR a11 + V0-matched p11/rarity; V1 left unchanged",
    }
    metrics_path.write_text(json.dumps(row, indent=2))
    decision = {
        "control": VARIANT,
        "question": "Does restoring joint prevalence + rarity recover V0 performance?",
        "metrics": row,
        "compare_to": {
            "V0_clean_valid": 0.7526,
            "V0_clean_test": 0.8219,
            "V1_clean_valid": 0.7457,
            "V1_clean_test": 0.5965,
        },
        "diagnostics": str(OUT / "audit" / "V1_vs_V0_CORR.json"),
    }
    (OUT / "audit" / "V1B_CLEAN_DECISION.json").write_text(json.dumps(decision, indent=2))
    print(json.dumps(row, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
