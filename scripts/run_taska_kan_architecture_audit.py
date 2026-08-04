#!/usr/bin/env python3
"""Wave KAN-ARCH — screen K1–K5 on clean + multihospital (seed 20260722).

Prerequisite: finish outputs/wave11_taskA/mlp_vs_kan_full (M0/K0 direct) OR pass --force-screen.
Conditional K6–K9 are stubs only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
WAVE_OUT = ROOT / "outputs" / "wave11_taskA"
OUT = WAVE_OUT / "kan_architecture_audit"
DIRECT_OUT = WAVE_OUT / "mlp_vs_kan_full"
DEFAULT_GSN = Path.home() / "Desktop" / "GSN Graphs dysertation 2026"
CFG_DIR = ROOT / "configs" / "taskA" / "kan_architectures"

SCREEN_VARIANTS = (
    "k1_shallow",
    "k2_mlp_to_kan",
    "k3_kan_to_linear",
    "k4_linear_plus_kan",
    "k5_grouped_kan",
)
SCREEN_SCENARIOS = ("clean", "multihospital")
SCREEN_SEED = 20260722
MULTISEED_SEEDS = (20260722, 20260723, 20260724)
EXPECTED_DIRECT = 36

# Arch keys promoted after Etap-1 screen (see SCREEN_PROMOTION.json).
DEFAULT_PROMOTED = ("k1_shallow", "k2_mlp_to_kan")

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


def direct_completed() -> int:
    return len(list(DIRECT_OUT.glob("runs/*/*/seed_*/metrics.json")))


def freeze_direct_decision() -> Path:
    """Write DIRECT_KAN_DECISION.json from completed direct runs (partial OK)."""
    rows = []
    for p in DIRECT_OUT.glob("runs/*/*/seed_*/metrics.json"):
        rows.append(json.loads(p.read_text()))
    by = {}
    for r in rows:
        by.setdefault(r["scenario"], {}).setdefault(r["model"], []).append(
            float(r["valid_auprc"])
        )
    deltas = {}
    for sc, models in by.items():
        if "TA_MLP" in models and "TA_KAN" in models:
            m = sum(models["TA_MLP"]) / len(models["TA_MLP"])
            k = sum(models["TA_KAN"]) / len(models["TA_KAN"])
            deltas[sc] = {"mlp_mean": m, "kan_mean": k, "delta": k - m}
    decision = {
        "wave": "TASKA_DIRECT_KAN",
        "n_runs": len(rows),
        "n_expected": EXPECTED_DIRECT,
        "complete": len(rows) >= EXPECTED_DIRECT,
        "scenario_deltas_valid_auprc": deltas,
        "reuse_as_M0_K0": True,
        "note": "M0=TA_MLP, K0=TA_KAN from outputs/wave11_taskA/mlp_vs_kan_full",
    }
    out = DIRECT_OUT / "DIRECT_KAN_DECISION.json"
    out.write_text(json.dumps(decision, indent=2))
    (OUT / "DIRECT_KAN_DECISION.json").write_text(json.dumps(decision, indent=2))
    return out


def build_cmd(cfg_path: Path, scenario: str, seed: int, epochs: int, run_dir: Path) -> list[str]:
    cfg = yaml.safe_load(cfg_path.read_text())
    pe = cfg["decoder"]["pair_encoder"]
    pe_type = str(pe["type"])
    cmd = [
        PY,
        str(ROOT / "src" / "train_taskA.py"),
        f"model={cfg['model']}",
        f"hcr={cfg['hcr']}",
        f"data.dataset.scenario={scenario}",
        "data.feature_ablation_profile=empirical",
        f"data.candidate_seed={seed}",
        f"training.seed={seed}",
        f"training.device={cfg['training'].get('device', 'cpu')}",
        f"training.epochs={epochs}",
        "training.early_stopping_patience=40",
        "training.grad_clip=1.0",
        "training.init_from_checkpoint=null",
        "experiment.wave=TASKA_KAN_ARCH",
        f"experiment.variant={cfg['variant']}",
        f"experiment.intervention={cfg['variant']}",
        "experiment.motif_completion.enabled=false",
        "model.decoder.arch=unshared_mlp",
        "model.decoder.ablation=none",
        "model.decoder.use_triple=false",
        "model.decoder.ag_kan_residual.enabled=false",
        f"model.decoder.pair_encoder.type={pe_type}",
        "model.decoder.pair_encoder.input_dim=40",
        "model.decoder.pair_encoder.hidden_dim=16",
        "model.decoder.pair_encoder.output_dim=8",
        f"model.decoder.pair_encoder.dropout={pe.get('dropout', 0.1)}",
        "model.decoder.pair_encoder.spline_order=3",
        "model.decoder.pair_encoder.grid_size=5",
        "model.decoder.pair_encoder.grid_range=[-3.0,3.0]",
        "model.decoder.pair_encoder.grid_update=false",
        "model.decoder.pair_encoder.base_activation=silu",
        f"model.decoder.pair_encoder.base_scale_init={pe.get('base_scale_init', 1.0)}",
        f"model.decoder.pair_encoder.spline_scale_init={pe.get('spline_scale_init', 0.1)}",
        "model.decoder.pair_encoder.use_bias=true",
        f"model.decoder.pair_encoder.spline_l1={pe.get('spline_l1', 1.0e-5)}",
        "wandb.enabled=false",
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
    if pe_type == "linear_plus_kan":
        cmd.append(
            f"model.decoder.pair_encoder.gate_init_logit={pe.get('gate_init_logit', -3.0)}"
        )
    return cmd


def _metrics_from_ckpt(ckpt: Path) -> dict:
    import torch

    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    fv = blob.get("final_valid_metrics") or {}
    ft = blob.get("final_test_metrics") or {}
    n_params = sum(int(v.numel()) for v in blob["model_state_dict"].values())
    return {
        "valid_auprc": float(fv.get("auprc", blob.get("best_valid_metric", float("nan")))),
        "test_auprc": float(ft.get("auprc", float("nan"))),
        "valid_auroc": float(fv.get("auc", fv.get("auroc", float("nan")))),
        "test_auroc": float(ft.get("auc", ft.get("auroc", float("nan")))),
        "valid_brier": float(fv.get("brier", float("nan"))),
        "test_brier": float(ft.get("brier", float("nan"))),
        "best_epoch": blob.get("best_epoch"),
        "parameter_count": n_params,
        "candidate_fingerprint": blob.get("candidate_fingerprint"),
    }


def run_one(
    arch_key: str,
    scenario: str,
    seed: int,
    epochs: int,
    *,
    stage: str = "screen",
) -> dict:
    cfg_path = CFG_DIR / f"{arch_key}.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    variant = cfg["variant"]
    run_dir = OUT / stage / "runs" / variant / scenario / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    ckpt = run_dir / "best_model.pt"
    if metrics_path.exists() and ckpt.exists():
        print(f"SKIP {variant} {scenario} seed={seed}", flush=True)
        return json.loads(metrics_path.read_text())

    # Reuse screen seed-22 checkpoint into multiseed folder when identical.
    if stage == "multiseed" and seed == SCREEN_SEED:
        alias = OUT / "screen" / "runs" / variant / scenario / f"seed_{seed}"
        if (alias / "metrics.json").exists() and (alias / "best_model.pt").exists():
            import shutil

            shutil.copy2(alias / "best_model.pt", ckpt)
            row = json.loads((alias / "metrics.json").read_text())
            row["reused_from"] = str(alias.relative_to(ROOT))
            metrics_path.write_text(json.dumps(row, indent=2, default=float))
            print(f"REUSE screen → multiseed {variant} {scenario} seed={seed}", flush=True)
            return row

    cmd = build_cmd(cfg_path, scenario, seed, epochs, run_dir)
    print("\nRUN:", " ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), env=_env())
    if proc.returncode != 0:
        raise SystemExit(f"failed {variant} {scenario} rc={proc.returncode}")
    row = {
        "architecture": variant,
        "arch_key": arch_key,
        "scenario": scenario,
        "seed": seed,
        "runtime_s": time.time() - t0,
        **_metrics_from_ckpt(ckpt),
    }
    metrics_path.write_text(json.dumps(row, indent=2, default=float))
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--force-screen", action="store_true")
    ap.add_argument("--freeze-direct-only", action="store_true")
    ap.add_argument(
        "--variants",
        default="all",
        help="Comma list of k1_shallow,... or all",
    )
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "screen").mkdir(exist_ok=True)
    (OUT / "diagnostics").mkdir(exist_ok=True)

    n = direct_completed()
    freeze_direct_decision()
    print(f"Direct M0/K0 runs completed: {n}/{EXPECTED_DIRECT}", flush=True)
    if args.freeze_direct_only:
        return
    if n < EXPECTED_DIRECT and not args.force_screen:
        raise SystemExit(
            f"Etap 0 incomplete ({n}/{EXPECTED_DIRECT}). "
            "Wait for taska_mlp_kan screen, or pass --force-screen."
        )

    variants = (
        list(SCREEN_VARIANTS)
        if args.variants == "all"
        else [v.strip() for v in args.variants.split(",") if v.strip()]
    )
    rows = []
    for scenario in SCREEN_SCENARIOS:
        for arch in variants:
            rows.append(run_one(arch, scenario, SCREEN_SEED, args.epochs))

    # summary vs MLP control on same scenario/seed if available
    mlp_ref = {}
    for p in DIRECT_OUT.glob(f"runs/TA_MLP/*/seed_{SCREEN_SEED}/metrics.json"):
        r = json.loads(p.read_text())
        mlp_ref[r["scenario"]] = float(r["valid_auprc"])

    sum_path = OUT / "architecture_summary.csv"
    with sum_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "architecture",
                "scenario",
                "seed",
                "valid_auprc",
                "test_auprc",
                "delta_vs_mlp",
                "parameter_count",
                "runtime_s",
                "promote_global_band",
            ],
        )
        w.writeheader()
        for r in rows:
            mlp = mlp_ref.get(r["scenario"])
            delta = (
                float(r["valid_auprc"]) - mlp if mlp is not None else float("nan")
            )
            promote = bool(delta == delta and delta >= -0.005)
            w.writerow(
                {
                    "architecture": r["architecture"],
                    "scenario": r["scenario"],
                    "seed": r["seed"],
                    "valid_auprc": r["valid_auprc"],
                    "test_auprc": r["test_auprc"],
                    "delta_vs_mlp": delta,
                    "parameter_count": r["parameter_count"],
                    "runtime_s": r["runtime_s"],
                    "promote_global_band": promote,
                }
            )

    decision = {
        "wave": "TASKA_KAN_ARCH_SCREEN",
        "seed": SCREEN_SEED,
        "scenarios": list(SCREEN_SCENARIOS),
        "variants": variants,
        "n_runs": len(rows),
        "next": "Promote variants with Δvalid≥−0.005 or endpoint-path nAUPRC gain; then multiseed.",
        "mlp_reference": mlp_ref,
    }
    (OUT / "FINAL_KAN_ARCHITECTURE_DECISION.json").write_text(
        json.dumps(decision, indent=2)
    )
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
