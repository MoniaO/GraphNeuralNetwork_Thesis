#!/usr/bin/env python3
"""Wave 7D Etap 3 — T0 (frozen A1) vs T1 (A1 + I(A;G|Z)).

Protocol
--------
- First seed 20260722 on clean.
- If T1 improves validation AUPRC over T0, repeat seeds 20260723/24.
- Selection on validation AUPRC only; test is report-only.
- No I(A;G) fallback when Z missing (triple = 0, mask = 0).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
OUT = ROOT / "outputs" / "wave7" / "wave7c" / "etap3"
DEFAULT_GSN = Path.home() / "Desktop" / "GSN Graphs dysertation 2026"


def _train_env() -> dict:
    import os

    env = os.environ.copy()
    env.setdefault("GSN_PROJECT_ROOT", str(DEFAULT_GSN))
    env.setdefault(
        "PHARMA_DATA_ROOT",
        str(Path(env["GSN_PROJECT_ROOT"]) / "2 v3. Data" / "dataset_v3"),
    )
    env.setdefault("PYTHONPATH", f"{ROOT / 'src'}:{ROOT}")
    env["PYTHONUNBUFFERED"] = "1"
    return env
SEEDS_FIRST = (20260722,)
SEEDS_CONFIRM = (20260722, 20260723, 20260724)

T0 = ("W7D_T0_A1_UNSHARED_ROLE_ENCODERS", "unshared_mlp", "none", False)
T1 = ("W7D_T1_A1_PLUS_CONDITIONAL_EDGE_GAIN", "unshared_mlp", "none", True)

# Reuse Etap-2 A1 seed-22 as T0.
T0_SEED22_ALIAS = (
    ROOT
    / "outputs/wave7/wave7c/etap2/runs/W7C_A1_UNSHARED_ROLE_ENCODERS/clean/seed_20260722"
)


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


def run_one(
    variant: str,
    arch: str,
    ablation: str,
    use_triple: bool,
    seed: int,
    epochs: int,
) -> dict:
    run_dir = OUT / "runs" / variant / "clean" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    ckpt = run_dir / "best_model.pt"
    if metrics_path.exists() and ckpt.exists():
        print(f"SKIP {variant} seed={seed}", flush=True)
        return json.loads(metrics_path.read_text())

    if (
        variant == T0[0]
        and seed == 20260722
        and (T0_SEED22_ALIAS / "best_model.pt").exists()
    ):
        print(f"REUSE etap2 A1 → {run_dir}", flush=True)
        shutil.copy2(T0_SEED22_ALIAS / "best_model.pt", ckpt)
        row = {
            "wave": "WAVE7D_ETAP3",
            "variant": variant,
            "arch": arch,
            "ablation": ablation,
            "use_triple": use_triple,
            "scenario": "clean",
            "seed": seed,
            "reused_from": str(T0_SEED22_ALIAS.relative_to(ROOT)),
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
        "data.dataset.scenario=clean",
        "data.feature_ablation_profile=empirical",
        f"data.candidate_seed={seed}",
        f"training.seed={seed}",
        "training.device=cpu",
        f"training.epochs={epochs}",
        "training.early_stopping_patience=40",
        "training.grad_clip=1.0",
        "experiment.wave=WAVE7D",
        f"experiment.variant={variant}",
        f"experiment.intervention={variant}",
        "experiment.motif_completion.enabled=false",
        f"model.decoder.arch={arch}",
        f"model.decoder.ablation={ablation}",
        f"model.decoder.use_triple={str(use_triple).lower()}",
        "wandb.enabled=true",
        "wandb.group=TaskA_WAVE7D_ETAP3_CMI",
        "wandb.job_type=wave7d_etap3",
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
        raise SystemExit(f"Failed {variant} seed={seed} rc={proc.returncode}")
    row = {
        "wave": "WAVE7D_ETAP3",
        "variant": variant,
        "arch": arch,
        "ablation": ablation,
        "use_triple": use_triple,
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
        return float("nan"), 0.0
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return m, math.sqrt(var)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument(
        "--force-three-seeds",
        action="store_true",
        help="Always run all three seeds even if T1 does not beat T0.",
    )
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    protocol = {
        "etap": 3,
        "T0": T0[0],
        "T1": T1[0],
        "base_arch": "unshared_mlp (A1 freeze from Etap 2)",
        "triple": "I(A;G|Z)=H(G|Z)-H(G|A,Z), binary-binary-binary only",
        "no_fallback_to_I_AG": True,
        "selection_metric": "valid_auprc",
        "seed_first": 20260722,
        "confirm_seeds_if_T1_wins": [20260723, 20260724],
    }
    (OUT / "ETAP3_PROTOCOL.json").write_text(json.dumps(protocol, indent=2))

    csv_path = OUT / "wave7d_etap3_results.csv"
    rows = []

    # Screen on seed 20260722.
    for variant, arch, ablation, use_triple in (T0, T1):
        row = run_one(variant, arch, ablation, use_triple, 20260722, args.epochs)
        append_csv(csv_path, row)
        rows.append(row)
        print(f"DONE {variant} seed=20260722: valid={row.get('valid_auprc')}", flush=True)

    by_v = {r["variant"]: r for r in rows if r["seed"] == 20260722}
    t0v = float(by_v[T0[0]]["valid_auprc"])
    t1v = float(by_v[T1[0]]["valid_auprc"])
    delta = t1v - t0v
    improves = delta > 0.0
    decision = {
        "seed_screen": 20260722,
        "T0_valid": t0v,
        "T1_valid": t1v,
        "T1_minus_T0": delta,
        "T1_improves_valid": improves,
        "action": (
            "confirm_three_seeds"
            if (improves or args.force_three_seeds)
            else "stop_after_screen"
        ),
    }
    print(json.dumps(decision, indent=2), flush=True)

    if improves or args.force_three_seeds:
        for seed in (20260723, 20260724):
            for variant, arch, ablation, use_triple in (T0, T1):
                row = run_one(variant, arch, ablation, use_triple, seed, args.epochs)
                append_csv(csv_path, row)
                rows.append(row)
                print(
                    f"DONE {variant} seed={seed}: valid={row.get('valid_auprc')}",
                    flush=True,
                )

    summary: dict = {"decision": decision, "by_variant": {}}
    for variant, _, _, _ in (T0, T1):
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
    if improves or args.force_three_seeds:
        summary["selected_for_etap4"] = (
            T1[0]
            if summary["by_variant"][T1[0]]["valid_auprc_mean"]
            >= summary["by_variant"][T0[0]]["valid_auprc_mean"]
            else T0[0]
        )
    else:
        summary["selected_for_etap4"] = T0[0]
        summary["note"] = "T1 did not improve valid AUPRC on seed 20260722; keep T0."
    (OUT / "ETAP3_SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
