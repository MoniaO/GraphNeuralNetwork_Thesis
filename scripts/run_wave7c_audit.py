#!/usr/bin/env python3
"""Wave 7C architecture + context audit screens (clean, seed 20260722).

Order: context coverage audit → A0–A3 → C0–C6.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
OUT = ROOT / "outputs" / "wave7" / "wave7c"
SEED = 20260722

ARCHITECTURE = [
    ("W7C_A0_R1_SHARED_PAIR_ENCODER", "shared_mlp", "none"),
    ("W7C_A1_UNSHARED_ROLE_ENCODERS", "unshared_mlp", "none"),
    ("W7C_A2_GLOBAL_MOTIF_ENCODER", "global_motif", "none"),
    ("W7C_A3_SHARED_LINEAR_PAIR_ENCODER", "shared_linear", "none"),
]

CONTEXT = [
    ("W7C_C0_FULL_MOTIF", "shared_mlp", "none"),
    ("W7C_C1_AG_ONLY", "shared_mlp", "ag_only"),
    ("W7C_C2_NO_DIRECT_AG", "shared_mlp", "no_direct_ag"),
    ("W7C_C3_MASK_TYPE_SUPPORT_ONLY", "shared_mlp", "mask_type_support_only"),
    ("W7C_C4_MATCHED_SHUFFLED_Z", "shared_mlp", "none"),
    ("W7C_C5_PATIENT_PERMUTED_DEPENDENCE", "shared_mlp", "none"),
    ("W7C_C6_EXPLICIT_ROLE_MASKS", "shared_mlp", "explicit_role_masks"),
]


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


def _subgroup_eval(ckpt: Path, variant: str, arch: str, ablation: str) -> dict:
    """Reload valid split and compute typed / context-subset AUPRC."""
    sys.path.insert(0, str(ROOT / "src"))
    import numpy as np
    import torch
    from hydra import compose, initialize_config_dir
    from sklearn.metrics import average_precision_score, brier_score_loss

    from data.PreprocessingTaskA.load_hetero_recon_data import load_recon_heterodata
    from hcr.wave7.panel_b.subgroup_metrics import pair_bucket, safe_auprc
    from hcr.wave7.wave7c.attach import fit_and_attach_wave7c
    from train_taskA import build_model

    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "model=TaskA_hgt_wave7c",
                "hcr=w7c_b2_audit",
                "data.dataset.scenario=clean",
                "data.feature_ablation_profile=empirical",
                f"data.candidate_seed={SEED}",
                f"training.seed={SEED}",
                "experiment.wave=WAVE7C",
                f"experiment.variant={variant}",
                f"model.decoder.arch={arch}",
                f"model.decoder.ablation={ablation}",
                "wandb.enabled=false",
            ],
        )
    train, valid, test, _ = load_recon_heterodata(cfg)
    fit_and_attach_wave7c(cfg, train, valid, test, device="cpu")
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = build_model(cfg, train)
    with torch.no_grad():
        _ = model(train)
    model.load_state_dict(blob["model_state_dict"])
    model.eval()
    with torch.no_grad():
        logits = model(valid).view(-1).cpu().numpy()
    y = valid.edge_label.detach().cpu().float().view(-1).numpy()
    p = 1.0 / (1.0 + np.exp(-logits))
    sources = list(valid.candidate_source_name)
    targets = list(valid.candidate_target_name)
    status = list(getattr(valid, "wave7c_context_status", ["unknown"] * len(y)))

    out: dict = {
        "valid_auprc_reload": safe_auprc(y, p),
        "valid_brier_reload": float(brier_score_loss(y, p)) if len(y) else float("nan"),
    }
    buckets = {k: [] for k in ("binary_binary", "count_involved", "continuous_involved", "mixed_type")}
    for u, v in zip(sources, targets):
        b = pair_bucket(str(u), str(v))
        for k in buckets:
            buckets[k].append(b[k])
    for k, flags in buckets.items():
        m = np.asarray(flags, dtype=bool)
        out[f"valid_auprc_{k}"] = safe_auprc(y[m], p[m]) if m.any() else float("nan")

    st = np.asarray(status)
    for name, mask in (
        ("valid_context", st == "valid"),
        ("missing_context", st == "missing"),
        ("matched_shuffled_context", st == "matched_shuffled"),
    ):
        if mask.any() and np.unique(y[mask]).size > 1:
            out[f"valid_auprc_{name}"] = float(average_precision_score(y[mask], p[mask]))
        else:
            out[f"valid_auprc_{name}"] = float("nan")
            out[f"n_{name}"] = int(mask.sum())
        out[f"n_{name}"] = int(mask.sum())
    return out


def run_one(variant: str, arch: str, ablation: str, epochs: int) -> dict:
    run_dir = OUT / "runs" / variant / "clean" / f"seed_{SEED}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    ckpt = run_dir / "best_model.pt"
    if metrics_path.exists() and ckpt.exists():
        print(f"SKIP {variant}", flush=True)
        return json.loads(metrics_path.read_text())

    t0 = time.time()
    cmd = [
        PY,
        str(ROOT / "src" / "train_taskA.py"),
        "model=TaskA_hgt_wave7c",
        "hcr=w7c_b2_audit",
        "data.dataset.scenario=clean",
        "data.feature_ablation_profile=empirical",
        f"data.candidate_seed={SEED}",
        f"training.seed={SEED}",
        "training.device=cpu",
        f"training.epochs={epochs}",
        "training.early_stopping_patience=40",
        "training.grad_clip=1.0",
        "experiment.wave=WAVE7C",
        f"experiment.variant={variant}",
        f"experiment.intervention={variant}",
        "experiment.motif_completion.enabled=false",
        f"model.decoder.arch={arch}",
        f"model.decoder.ablation={ablation}",
        "wandb.enabled=true",
        "wandb.group=TaskA_WAVE7C_CONTEXT_AUDIT",
        "wandb.job_type=wave7c_audit",
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
    runtime_s = time.time() - t0
    if proc.returncode != 0:
        raise SystemExit(f"Failed {variant} rc={proc.returncode}")
    row = {
        "wave": "WAVE7C",
        "variant": variant,
        "arch": arch,
        "ablation": ablation,
        "scenario": "clean",
        "seed": SEED,
        "runtime_s": runtime_s,
        **_metrics_from_ckpt(ckpt),
    }
    try:
        row.update(_subgroup_eval(ckpt, variant, arch, ablation))
    except Exception as exc:  # noqa: BLE001 — audit must not fail the suite
        row["subgroup_eval_error"] = str(exc)
        print(f"WARN subgroup eval failed for {variant}: {exc}", flush=True)
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


def _delta(a: dict, b: dict | None):
    if not a or not b:
        return None
    return float(a["valid_auprc"]) - float(b["valid_auprc"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--skip-audit", action="store_true")
    ap.add_argument("--only", default="all", help="arch|context|all")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if not args.skip_audit:
        print("Running context coverage audit…", flush=True)
        proc = subprocess.run(
            [PY, str(ROOT / "scripts" / "audit_wave7c_context_coverage.py")],
            cwd=str(ROOT),
        )
        if proc.returncode != 0:
            raise SystemExit(f"audit failed rc={proc.returncode}")

    variants = []
    if args.only in {"all", "arch"}:
        variants.extend(ARCHITECTURE)
    if args.only in {"all", "context"}:
        variants.extend(CONTEXT)

    csv_path = OUT / "wave7c_results.csv"
    rows = []
    for variant, arch, ablation in variants:
        row = run_one(variant, arch, ablation, args.epochs)
        append_csv(csv_path, row)
        rows.append(row)
        print(f"DONE {variant}: valid={row.get('valid_auprc')}", flush=True)

    by_v = {r["variant"]: r for r in rows}
    a0 = by_v.get("W7C_A0_R1_SHARED_PAIR_ENCODER", {})
    c0 = by_v.get("W7C_C0_FULL_MOTIF", a0)
    decision = {
        "selection_metric": "valid_auprc",
        "A0_valid": a0.get("valid_auprc"),
        "C0_valid": c0.get("valid_auprc"),
        "deltas_arch": {
            "A0_minus_A1": _delta(a0, by_v.get("W7C_A1_UNSHARED_ROLE_ENCODERS")),
            "A0_minus_A2": _delta(a0, by_v.get("W7C_A2_GLOBAL_MOTIF_ENCODER")),
            "A0_minus_A3": _delta(a0, by_v.get("W7C_A3_SHARED_LINEAR_PAIR_ENCODER")),
        },
        "deltas_context": {
            "C0_minus_C1": _delta(c0, by_v.get("W7C_C1_AG_ONLY")),
            "C0_minus_C2": _delta(c0, by_v.get("W7C_C2_NO_DIRECT_AG")),
            "C0_minus_C3": _delta(c0, by_v.get("W7C_C3_MASK_TYPE_SUPPORT_ONLY")),
            "C0_minus_C4": _delta(c0, by_v.get("W7C_C4_MATCHED_SHUFFLED_Z")),
            "C0_minus_C5": _delta(c0, by_v.get("W7C_C5_PATIENT_PERMUTED_DEPENDENCE")),
            "C0_minus_C6": _delta(c0, by_v.get("W7C_C6_EXPLICIT_ROLE_MASKS")),
        },
        "desired": "C0>C1,C2,C4,C5 and C3≪C0; A0 vs A1/A2/A3 for inductive bias",
        "rows": rows,
    }
    (OUT / "WAVE7C_DECISION.json").write_text(json.dumps(decision, indent=2))
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
