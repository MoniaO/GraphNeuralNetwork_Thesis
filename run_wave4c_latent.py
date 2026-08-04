#!/usr/bin/env python3
"""Wave 4C latent-gate matrix — robust Python runner (no fragile bash)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

DAY = datetime.now().strftime("%Y-%m-%d")
OUT_DIR = ROOT / f"outputs/wave4c_latent_{DAY}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG = OUT_DIR / f"{DAY}_wave4c_LATENT_VISIBLE.log"
CSV = OUT_DIR / f"{DAY}_wave4c_LATENT_all_runs.csv"

SEEDS = [20260721, 20260722, 20260723]
VARIANTS = [
    "none",
    "latent_pairwise_aby",
    "hcr3_full",
    "hcr3_without_a111",
    "hcr3_shuffled",
    "hcr3_random_context",
]

ENV = os.environ.copy()
ENV["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT}:{ENV.get('PYTHONPATH', '')}"
ENV["PYTHONUNBUFFERED"] = "1"
ENV.setdefault(
    "GSN_PROJECT_ROOT",
    str(Path.home() / "Desktop" / "GSN Graphs dysertation 2026"),
)
ENV.setdefault(
    "PHARMA_DATA_ROOT",
    str(Path(ENV["GSN_PROJECT_ROOT"]) / "2 v3. Data" / "dataset_v3"),
)


def log(msg: str) -> None:
    line = msg if msg.endswith("\n") else msg + "\n"
    with LOG.open("a") as f:
        f.write(line)
    print(line, end="", flush=True)


def already_done(variant: str, seed: int) -> bool:
    p = OUT_DIR / f"latent_metrics_{variant}_{seed}.json"
    return p.exists()


def newest_ckpt_after(t0: float) -> Path | None:
    best = None
    best_m = t0
    for p in (ROOT / "outputs").glob("*/**/best_model.pt"):
        try:
            m = p.stat().st_mtime
        except OSError:
            continue
        if m >= t0 and m >= best_m:
            best, best_m = p, m
    # also flat hydra layout outputs/YYYY-MM-DD/HH-MM-SS/
    for p in (ROOT / "outputs").glob("*/*/best_model.pt"):
        try:
            m = p.stat().st_mtime
        except OSError:
            continue
        if m >= t0 and m >= best_m:
            best, best_m = p, m
    return best


def train_cmd(variant: str, seed: int) -> list[str]:
    if variant == "none":
        model = ["model=TaskA_hgt", "hcr=none"]
    else:
        model = ["model=TaskA_hgt_hcr", f"hcr={variant}"]
    zmode = []
    if variant == "hcr3_random_context":
        zmode = ["hcr.z_mode=random_matched"]
    elif variant.startswith("hcr3_") or variant == "latent_pairwise_aby":
        zmode = ["hcr.z_mode=oracle_outcome"]
    return [
        sys.executable,
        "src/train_taskA.py",
        *model,
        *zmode,
        "model.num_layers=1",
        "model.hidden_dim=64",
        "model.hidden_channels=64",
        "model.heads=8",
        "model.dropout=0.2",
        "data.dataset.scenario=clean",
        "data.feature_ablation_profile=empirical",
        "data.candidate_seed=20260722",
        f"training.seed={seed}",
        "training.device=cpu",
        "experiment.wave=WAVE4C_LATENT",
        "experiment.motif_completion.enabled=true",
        "experiment.motif_completion.hide=parent_a",
        "experiment.latent_gate.enabled=true",
        f"experiment.intervention=latent_gate__{variant}",
        f"experiment.hcr_variant={variant}",
        "wandb.enabled=false",
    ]


def append_csv(variant: str, seed: int, ckpt: Path, metrics: dict) -> None:
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    fv = payload.get("final_valid_metrics") or {}
    ft = payload.get("final_test_metrics") or {}
    row = {
        "wave": "wave4c_latent",
        "hcr_variant": variant,
        "training_seed": int(seed),
        "valid_auprc": fv.get("auprc", payload.get("best_valid_metric")),
        "valid_brier": fv.get("brier"),
        "test_auprc": ft.get("auprc"),
        "test_brier": ft.get("brier"),
        "best_epoch": payload.get("best_epoch"),
        "checkpoint_path": str(ckpt),
        "auprc_motif": metrics.get("auprc_motif_heldout_vs_valid_neg"),
        "mean_prob_motif": metrics.get("mean_prob_motif"),
        "n_motif_positives": metrics.get("n_motif_positives"),
    }
    df = pd.DataFrame([row])
    if CSV.exists():
        old = pd.read_csv(CSV)
        df = pd.concat([old, df], ignore_index=True).drop_duplicates(
            ["hcr_variant", "training_seed"], keep="last"
        )
    df.to_csv(CSV, index=False)


def main() -> None:
    log("=" * 60)
    log("WAVE 4C — LATENT GATE RECOVERY (python runner)")
    log(f"Variants: {VARIANTS}")
    log(f"Seeds: {SEEDS}")
    log(f"Started: {datetime.now()}")
    log("=" * 60)

    for seed in SEEDS:
        for variant in VARIANTS:
            if already_done(variant, seed):
                log(f"SKIP {variant} seed={seed}")
                continue
            log(f"===== START {variant} seed={seed} {datetime.now():%H:%M:%S} =====")
            t0 = time.time()
            with LOG.open("a") as lf:
                proc = subprocess.run(
                    train_cmd(variant, seed),
                    cwd=str(ROOT),
                    env=ENV,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                )
            rc = proc.returncode
            log(f"===== DONE {variant} seed={seed} rc={rc} {datetime.now():%H:%M:%S} =====")
            if rc != 0:
                log(f"FAIL train {variant} seed={seed}")
                continue
            ckpt = newest_ckpt_after(t0 - 1.0)
            if ckpt is None or not ckpt.exists():
                log(f"WARN: no checkpoint for {variant} seed={seed}")
                continue
            out_json = OUT_DIR / f"latent_metrics_{variant}_{seed}.json"
            ev = subprocess.run(
                [
                    sys.executable,
                    "scripts/eval_wave4c_latent_gate.py",
                    "--checkpoint",
                    str(ckpt),
                    "--hcr",
                    variant,
                    "--seed",
                    str(seed),
                    "--out",
                    str(out_json),
                ],
                cwd=str(ROOT),
                env=ENV,
                capture_output=True,
                text=True,
            )
            with LOG.open("a") as lf:
                lf.write(ev.stdout or "")
                lf.write(ev.stderr or "")
            if ev.returncode != 0 or not out_json.exists():
                log(f"WARN: eval failed {variant} seed={seed}")
                continue
            metrics = json.loads(out_json.read_text())
            append_csv(variant, seed, ckpt, metrics)
            log(
                f"  motif_auprc={metrics.get('auprc_motif_heldout_vs_valid_neg'):.4f} "
                f"ckpt={ckpt}"
            )

    # summary
    rows = []
    for p in sorted(OUT_DIR.glob("latent_metrics_*.json")):
        name = p.stem.replace("latent_metrics_", "")
        *vp, seed = name.rsplit("_", 1)
        m = json.loads(p.read_text())
        rows.append(
            {
                "variant": "_".join(vp),
                "seed": int(seed),
                "auprc": m["auprc_motif_heldout_vs_valid_neg"],
            }
        )
    if rows:
        df = pd.DataFrame(rows)
        summary = (
            df.groupby("variant")["auprc"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values("mean", ascending=False)
        )
        summary.to_csv(OUT_DIR / f"{DAY}_wave4c_LATENT_summary.csv", index=False)
        log("\n=== Wave 4C motif AUPRC ===")
        log(summary.to_string(index=False))
    (OUT_DIR / "PROTOCOL.txt").write_text(
        "Wave 4C — Latent-gate HCR recovery\n"
        "Hide A→G; features use (A,Y,B) never gate G.\n"
        "Primary metric: auprc_motif_heldout_vs_valid_neg\n"
        "Wave 5 PARKED until Wave 4 frozen.\n"
    )
    log(f"WAVE4C finished {datetime.now()} → {OUT_DIR}")


if __name__ == "__main__":
    main()
