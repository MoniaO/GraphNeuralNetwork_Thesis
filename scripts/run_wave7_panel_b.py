#!/usr/bin/env python3
"""Wave 7 Panel B — jitter + Legendre GHCR (clean audit first).

Variants
--------
W7B_B0_LEGACY_V0_PAD40
W7B_B1_JITTER_GHCR_MATRIX40
W7B_B2_JITTER_GHCR_ENRICHED40
W7B_B3_HYBRID_LEGACY_BINARY_JITTER40

Protocol
--------
- seed 20260722
- clean only before any other scenario
- HCR fit train-only; no bootstrap in the vector
- Motif AZ⊕AG⊕ZG → 120-d
- Select on validation AUPRC (not test)
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
OUT = ROOT / "outputs" / "wave7" / "panel_b"
SEED = 20260722

VARIANTS = [
    ("W7B_B0_LEGACY_V0_PAD40", "w7b_b0_legacy_v0_pad40"),
    ("W7B_B1_JITTER_GHCR_MATRIX40", "w7b_b1_jitter_ghcr_matrix40"),
    ("W7B_B2_JITTER_GHCR_ENRICHED40", "w7b_b2_jitter_ghcr_enriched40"),
    ("W7B_B3_HYBRID_LEGACY_BINARY_JITTER40", "w7b_b3_hybrid_legacy_binary_jitter40"),
]


def _metrics_from_ckpt(ckpt: Path) -> dict:
    import torch

    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    final_test = (
        blob.get("final_test_metrics")
        or blob.get("final_test")
        or blob.get("test")
        or {}
    )
    final_valid = (
        blob.get("final_valid_metrics")
        or blob.get("final_valid")
        or blob.get("valid")
        or {}
    )
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


def run_one(variant: str, hcr_name: str, scenario: str, epochs: int) -> dict:
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
        "model=TaskA_hgt_wave7b",
        f"hcr={hcr_name}",
        f"data.dataset.scenario={scenario}",
        "data.feature_ablation_profile=empirical",
        f"data.candidate_seed={SEED}",
        f"training.seed={SEED}",
        "training.device=cpu",
        f"training.epochs={epochs}",
        "training.early_stopping_patience=40",
        "experiment.wave=WAVE7_PANEL_B",
        f"experiment.variant={variant}",
        f"experiment.intervention={variant}",
        "experiment.motif_completion.enabled=false",
        "wandb.enabled=true",
        "wandb.group=TaskA_WAVE7_PANEL_B_JITTER",
        "wandb.job_type=wave7_panel_b",
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
        "model.decoder.hcr_dim=120",
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
        "wave": "WAVE7_PANEL_B",
        "variant": variant,
        "scenario": scenario,
        "seed": SEED,
        "arch_tag": "joint_L3_H8_W32_leaky_relu",
        "hcr_dim": 120,
        **_metrics_from_ckpt(ckpt),
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
    ap.add_argument(
        "--scenarios",
        default="clean",
        help="Comma-separated; default clean-only audit.",
    )
    ap.add_argument(
        "--variants",
        default="all",
        help="Comma-separated variant names or 'all'.",
    )
    args = ap.parse_args()
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    if args.variants == "all":
        variants = VARIANTS
    else:
        wanted = {v.strip() for v in args.variants.split(",") if v.strip()}
        variants = [v for v in VARIANTS if v[0] in wanted or v[1] in wanted]

    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        "wave": "WAVE7_PANEL_B",
        "wandb_group": "TaskA_WAVE7_PANEL_B_JITTER",
        "seed": SEED,
        "scenarios_first": scenarios,
        "variants": [v[0] for v in variants],
        "motif": "AZ⊕AG⊕ZG",
        "pair_dim": 40,
        "motif_dim": 120,
        "bootstrap_in_vector": False,
        "selection": "validation AUPRC (not test)",
    }
    (OUT / "WAVE7_PANEL_B_PROTOCOL.json").write_text(json.dumps(protocol, indent=2))

    csv_path = OUT / "wave7_panel_b_results.csv"
    rows = []
    for scenario in scenarios:
        for variant, hcr_name in variants:
            row = run_one(variant, hcr_name, scenario, args.epochs)
            append_csv(csv_path, row)
            rows.append(row)
            print(
                f"DONE {variant} {scenario}: "
                f"valid_auprc={row.get('valid_auprc')} "
                f"test_auprc={row.get('test_auprc')}",
                flush=True,
            )

    if rows:
        # Select on validation only.
        clean = [r for r in rows if r.get("scenario") == "clean"]
        if clean:
            best = max(clean, key=lambda r: float(r.get("valid_auprc", float("-inf"))))
            decision = {
                "selection_metric": "valid_auprc",
                "selected_variant": best["variant"],
                "selected_valid_auprc": best.get("valid_auprc"),
                "selected_test_auprc_not_for_selection": best.get("test_auprc"),
                "note": "Do not choose variants using test AUPRC.",
                "rows": clean,
            }
            (OUT / "WAVE7_PANEL_B_DECISION.json").write_text(json.dumps(decision, indent=2))
            print(
                f"\nSELECTED (valid AUPRC): {best['variant']} "
                f"valid={best.get('valid_auprc')} test(report-only)={best.get('test_auprc')}",
                flush=True,
            )


if __name__ == "__main__":
    main()
