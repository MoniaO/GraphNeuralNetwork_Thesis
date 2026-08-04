#!/usr/bin/env python3
"""Wave 7 — generalized HCR variants on frozen L3_H8_W32_leaky architecture.

Variants
--------
W7_V0_L2_BINARY_COMPACT
W7_V1_GHCR_BINARY_ONEHOT
W7_V2_GHCR_ALL_TYPES_ONEHOT_COUNT

Protocol
--------
- seed 20260722 (training + candidates)
- first: three clean pilots
- then: all patient scenarios × three variants
- HCR fit train-only
- Shared frozen motif: AZ⊕AG⊕ZG (candidate_coparent_triangle), NOT AB⊕AY⊕BY
  with Y=downstream. Same Wave 5C registry / triples for V0/V1/V2.
- Dual/triple contexts = expert_gate_schema; others = G_train parents.
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
OUT = ROOT / "outputs" / "wave7"
SEED = 20260722

VARIANTS = [
    ("W7_V0_L2_BINARY_COMPACT", "w7_v0_l2_binary_compact"),
    ("W7_V1_GHCR_BINARY_ONEHOT", "w7_v1_ghcr_binary_onehot"),
    ("W7_V2_GHCR_ALL_TYPES_ONEHOT_COUNT", "w7_v2_ghcr_all_types_onehot_count"),
]

SCENARIOS = [
    "clean",
    "hidden_confounder",
    "selection_bias",
    "no_overlap",
    "noisy_documentation",
    "multihospital",
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
    if metrics_path.exists():
        print(f"SKIP {variant} {scenario}", flush=True)
        return json.loads(metrics_path.read_text())

    cmd = [
        PY,
        str(ROOT / "src" / "train_taskA.py"),
        "model=TaskA_hgt_wave7",
        f"hcr={hcr_name}",
        f"data.dataset.scenario={scenario}",
        "data.feature_ablation_profile=empirical",
        f"data.candidate_seed={SEED}",
        f"training.seed={SEED}",
        "training.device=cpu",
        f"training.epochs={epochs}",
        "training.early_stopping_patience=40",
        "experiment.wave=WAVE7",
        f"experiment.variant={variant}",
        f"experiment.intervention={variant}",
        "experiment.motif_completion.enabled=false",
        "wandb.enabled=true",
        "wandb.group=TaskA_WAVE7_GHCR",
        "wandb.job_type=wave7_variant",
        f"hydra.run.dir={run_dir}",
        # freeze arch explicitly (also baked into TaskA_hgt_wave7.yaml)
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
    print("\nRUN:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        raise SystemExit(f"Failed {variant} {scenario} rc={proc.returncode}")
    if not ckpt.exists():
        raise SystemExit(f"Missing checkpoint {ckpt}")
    row = {
        "wave": "WAVE7",
        "variant": variant,
        "scenario": scenario,
        "seed": SEED,
        "arch_tag": "joint_L3_H8_W32_leaky_relu",
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


def upload_summary(csv_path: Path) -> None:
    try:
        import pandas as pd
        import wandb
    except ImportError:
        print("wandb/pandas missing — skip summary upload")
        return
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    wandb.init(
        project="politechnika-gnn-thesis",
        entity="politechnika-gnn-thesis",
        group="TaskA_WAVE7_GHCR",
        job_type="wave7_summary",
        name="wave7_ghcr_summary",
        reinit=True,
    )
    wandb.log({"wave7_results": wandb.Table(dataframe=df)})
    wandb.summary["n_rows"] = len(df)
    if len(df):
        best = df.sort_values("valid_auprc", ascending=False).iloc[0]
        wandb.summary["top_variant"] = str(best["variant"])
        wandb.summary["top_scenario"] = str(best["scenario"])
        wandb.summary["top_valid_auprc"] = float(best["valid_auprc"])
    wandb.finish()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage",
        choices=["pilots", "scenarios", "all"],
        default="pilots",
        help="pilots=3 clean variants; scenarios=all scenario×variant; all=both",
    )
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "wave7_results.csv"
    # freeze pointer
    frozen = {
        "arch_tag": "joint_L3_H8_W32_leaky_relu",
        "benchmark_wave1_note": "Wave 1 HeteroSAGE/R-GCN kept as historical baseline; Wave 7 uses frozen L2-audit winner.",
        "seed": SEED,
        "variants": [v for v, _ in VARIANTS],
        "scenarios": SCENARIOS,
    }
    (OUT / "WAVE7_PROTOCOL.json").write_text(json.dumps(frozen, indent=2))

    jobs: list[tuple[str, str, str]] = []
    if args.stage == "pilots":
        scenarios = ["clean"]
    else:
        # scenarios | all → full patient-scenario matrix
        scenarios = list(SCENARIOS)
    for scenario in scenarios:
        for variant, hcr in VARIANTS:
            jobs.append((variant, hcr, scenario))
    uniq = jobs

    rows = []
    for variant, hcr, scenario in uniq:
        row = run_one(variant, hcr, scenario, args.epochs)
        append_csv(csv_path, row)
        rows.append(row)
        print(
            f"DONE {variant} {scenario}: valid={row['valid_auprc']:.4f} "
            f"test={row['test_auprc']:.4f}",
            flush=True,
        )

    # Rebuild clean summary CSV from all metrics.json (resume-safe)
    all_rows = []
    for metrics in (OUT / "runs").glob("*/*/seed_*/metrics.json"):
        all_rows.append(json.loads(metrics.read_text()))
    if all_rows:
        import pandas as pd

        df = pd.DataFrame(all_rows)
        df.to_csv(csv_path, index=False)
        summary = (
            df.groupby(["variant", "scenario"], as_index=False)
            .agg(
                n=("seed", "count"),
                valid_auprc=("valid_auprc", "mean"),
                test_auprc=("test_auprc", "mean"),
                test_brier=("test_brier", "mean"),
            )
            .sort_values(["scenario", "valid_auprc"], ascending=[True, False])
        )
        summary.to_csv(OUT / "wave7_summary.csv", index=False)
        (OUT / "WAVE7_DECISION.json").write_text(
            json.dumps(
                {
                    "wave": "WAVE7",
                    "seed": SEED,
                    "frozen_architecture": "joint_L3_H8_W32_leaky_relu",
                    "n_runs": int(len(df)),
                    "clean_leaderboard": summary[summary.scenario == "clean"].to_dict(
                        orient="records"
                    ),
                },
                indent=2,
            )
        )
        upload_summary(csv_path)
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
