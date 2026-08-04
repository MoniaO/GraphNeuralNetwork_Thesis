#!/usr/bin/env python3
"""Wave 4D — Causal Context Selection & Role Audit (36 runs)."""

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

DAY = "2026-07-31"
OUT_DIR = ROOT / f"outputs/wave4d_context_role_audit_{DAY}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG = OUT_DIR / f"{DAY}_wave4d_CONTEXT_ROLE_AUDIT.log"
CSV = OUT_DIR / f"{DAY}_wave4d_all_runs.csv"

SEEDS = [20260721, 20260722, 20260723]
MASKS = ["parent_a", "parent_b"]
VARIANTS = [
    "none",
    "structural_latent_pairwise",
    "all_context_top1",
    "structural_context_shuffled",
    "matched_random_context",
    "hcr3_selected_capacity_matched",
]

VARIANT_MODE = {
    "none": "none",
    "structural_latent_pairwise": "structural",
    "all_context_top1": "all_context_top1",
    "structural_context_shuffled": "structural",
    "matched_random_context": "matched_random",
    "hcr3_selected_capacity_matched": "structural",
}

PYTHON = str(ROOT / ".venv" / "bin" / "python")
if not Path(PYTHON).exists():
    PYTHON = sys.executable

ENV = os.environ.copy()
ENV["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT}:{ENV.get('PYTHONPATH', '')}"
ENV["PYTHONUNBUFFERED"] = "1"
# Do not force offline — Wave 4D should appear under
# wandb group TaskA_WAVE4D_CONTEXT_ROLE_AUDIT.
ENV.pop("WANDB_MODE", None) if ENV.get("WANDB_MODE") == "offline" else None
ENV["WANDB_MODE"] = "online"
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


def key(variant: str, seed: int, hide: str) -> str:
    return f"{variant}__{hide}__{seed}"


def already_done(variant: str, seed: int, hide: str) -> bool:
    return (OUT_DIR / f"wave4d_metrics_{key(variant, seed, hide)}.json").exists()


def newest_ckpt_after(t0: float) -> Path | None:
    best = None
    best_m = t0
    for pattern in ("*/**/best_model.pt", "*/*/best_model.pt"):
        for p in (ROOT / "outputs").glob(pattern):
            try:
                m = p.stat().st_mtime
            except OSError:
                continue
            if m >= t0 and m >= best_m:
                best, best_m = p, m
    return best


def train_cmd(variant: str, seed: int, hide: str) -> list[str]:
    if variant == "none":
        model = ["model=TaskA_hgt", "hcr=none"]
    else:
        model = ["model=TaskA_hgt_hcr", f"hcr={variant}"]
    return [
        PYTHON,
        "src/train_taskA.py",
        *model,
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
        "experiment.wave=WAVE4D_CONTEXT_ROLE_AUDIT",
        "experiment.motif_completion.enabled=true",
        f"experiment.motif_completion.hide={hide}",
        "experiment.causal_role_audit.enabled=true",
        f"experiment.context_selection.mode={VARIANT_MODE[variant]}",
        "experiment.context_selection.source_graph=train",
        "experiment.context_selection.use_true_graph=false",
        f"experiment.intervention=wave4d__{variant}__{hide}",
        f"experiment.hcr_variant={variant}",
        # Offline by default (WANDB_MODE=offline); group preserved for sync.
        "wandb.enabled=true",
        "wandb.group=TaskA_WAVE4D_CONTEXT_ROLE_AUDIT",
        f"wandb.tags=[TaskA,WAVE4D,{variant},{hide}]",
        "wandb.job_type=training",
    ]


def append_csv(variant: str, seed: int, hide: str, ckpt: Path, metrics: dict) -> None:
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    fv = payload.get("final_valid_metrics") or {}
    ft = payload.get("final_test_metrics") or {}
    pa = metrics.get("panel_a") or {}
    pb = metrics.get("panel_b") or {}
    row = {
        "wave": "wave4d_context_role_audit",
        "hcr_variant": variant,
        "hide": hide,
        "training_seed": int(seed),
        "valid_auprc": fv.get("auprc", payload.get("best_valid_metric")),
        "test_auprc": ft.get("auprc"),
        "best_epoch": payload.get("best_epoch"),
        "checkpoint_path": str(ckpt),
        "auprc_motif": pa.get("auprc_motif"),
        "mrr": pa.get("mrr"),
        "hits_at_1": pa.get("hits_at_1"),
        "hits_at_3": pa.get("hits_at_3"),
        "hits_at_5": pa.get("hits_at_5"),
        "mean_rank": pa.get("mean_rank"),
        "median_rank": pa.get("median_rank"),
        "context_selection_precision_at_1": metrics.get("context_selection_precision_at_1"),
        "fpr_confounder": (pb.get("confounder") or {}).get("fpr_at_0.5"),
        "fpr_mediator": (pb.get("mediator") or {}).get("fpr_at_0.5"),
        "fpr_collider": (pb.get("collider") or {}).get("fpr_at_0.5"),
        "fpr_descendant": (pb.get("descendant") or {}).get("fpr_at_0.5"),
    }
    df = pd.DataFrame([row])
    if CSV.exists():
        old = pd.read_csv(CSV)
        df = pd.concat([old, df], ignore_index=True).drop_duplicates(
            ["hcr_variant", "hide", "training_seed"], keep="last"
        )
    df.to_csv(CSV, index=False)


def main() -> None:
    log("=" * 60)
    log("WAVE 4D — CAUSAL CONTEXT SELECTION AND ROLE AUDIT")
    log(f"Variants: {VARIANTS}")
    log(f"Seeds: {SEEDS}")
    log(f"Masks: {MASKS}")
    log(f"Matrix: {len(VARIANTS)}×{len(SEEDS)}×{len(MASKS)} = {len(VARIANTS)*len(SEEDS)*len(MASKS)}")
    log(f"Started: {datetime.now()}")
    log("=" * 60)

    for hide in MASKS:
        for seed in SEEDS:
            for variant in VARIANTS:
                tag = key(variant, seed, hide)
                if already_done(variant, seed, hide):
                    log(f"SKIP {tag}")
                    continue
                log(f"===== START {tag} {datetime.now():%H:%M:%S} =====")
                t0 = time.time()
                with LOG.open("a") as lf:
                    proc = subprocess.run(
                        train_cmd(variant, seed, hide),
                        cwd=str(ROOT),
                        env=ENV,
                        stdout=lf,
                        stderr=subprocess.STDOUT,
                    )
                rc = proc.returncode
                log(f"===== DONE {tag} rc={rc} {datetime.now():%H:%M:%S} =====")
                if rc != 0:
                    log(f"FAIL train {tag}")
                    continue
                ckpt = newest_ckpt_after(t0 - 1.0)
                if ckpt is None or not ckpt.exists():
                    log(f"WARN: no checkpoint for {tag}")
                    continue
                out_json = OUT_DIR / f"wave4d_metrics_{tag}.json"
                ev = subprocess.run(
                [
                    PYTHON,
                    "scripts/eval_wave4d_context_role.py",
                        "--checkpoint",
                        str(ckpt),
                        "--hcr",
                        variant,
                        "--seed",
                        str(seed),
                        "--hide",
                        hide,
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
                    log(f"WARN: eval failed {tag}\n{ev.stderr}")
                    continue
                metrics = json.loads(out_json.read_text())
                append_csv(variant, seed, hide, ckpt, metrics)
                pa = metrics.get("panel_a") or {}
                log(
                    f"  motif_auprc={pa.get('auprc_motif')} "
                    f"prec@1={metrics.get('context_selection_precision_at_1')} "
                    f"ckpt={ckpt}"
                )

    # Summaries
    rows = []
    for p in sorted(OUT_DIR.glob("wave4d_metrics_*.json")):
        m = json.loads(p.read_text())
        pa = m.get("panel_a") or {}
        rows.append(
            {
                "variant": m["hcr"],
                "hide": m["hide"],
                "seed": m["seed"],
                "auprc_motif": pa.get("auprc_motif"),
                "mrr": pa.get("mrr"),
                "hits_at_1": pa.get("hits_at_1"),
                "precision_at_1": m.get("context_selection_precision_at_1"),
                "fpr_collider": (m.get("panel_b") or {}).get("collider", {}).get("fpr_at_0.5"),
                "fpr_descendant": (m.get("panel_b") or {}).get("descendant", {}).get("fpr_at_0.5"),
            }
        )
    if rows:
        df = pd.DataFrame(rows)
        summary = (
            df.groupby(["variant", "hide"])
            .agg(
                auprc_mean=("auprc_motif", "mean"),
                auprc_std=("auprc_motif", "std"),
                mrr_mean=("mrr", "mean"),
                prec1_mean=("precision_at_1", "mean"),
                fpr_collider_mean=("fpr_collider", "mean"),
                fpr_descendant_mean=("fpr_descendant", "mean"),
                n=("seed", "count"),
            )
            .reset_index()
            .sort_values(["hide", "auprc_mean"], ascending=[True, False])
        )
        summary.to_csv(OUT_DIR / f"{DAY}_wave4d_summary.csv", index=False)
        log("\n=== Wave 4D summary ===")
        log(summary.to_string(index=False))

    (OUT_DIR / "PROTOCOL.txt").write_text(
        "Wave 4D — Causal Context Selection and Role Audit\n"
        "W&B group: TaskA_WAVE4D_CONTEXT_ROLE_AUDIT\n"
        "Question: Can context be chosen without scanning all columns,\n"
        "and without confusing co-parent vs mediator/confounder/collider/descendant?\n"
        "Masks: parent_a, parent_b. Variants D0–D5. Seeds 20260721–23.\n"
        "Final candidate: D1 structural_latent_pairwise (not D2).\n"
        "G_true is evaluator-only for role registry.\n"
        "Freeze for Wave 5 if D1 beats D0/D3/D4 and is safer than D2.\n"
    )
    log(f"WAVE4D finished {datetime.now()} → {OUT_DIR}")


if __name__ == "__main__":
    main()
