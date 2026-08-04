#!/usr/bin/env python3
"""Wave 10 — fair residual compare.

R0 = eval-only reload of the frozen A1 checkpoint (NO retrain).
R1 = same A1 checkpoint + AG KAN residual (HGT/MLP/decoder frozen).

Canonical backbone:
  outputs/wave7/wave7c/etap2/runs/W7C_A1_UNSHARED_ROLE_ENCODERS/clean/seed_20260722/best_model.pt

Writes:
  outputs/wave10/audits/W10_R0_BASELINE_AUDIT.json
  outputs/wave10/summaries/wave10_fair_screen.csv
  outputs/wave10/summaries/WAVE10_FAIR_DECISION.json
  outputs/wave10/runs/W10_R0_EVAL_A1/.../metrics.json
  outputs/wave10/runs/W10_R1_AG_KAN_RESIDUAL_FROM_A1/.../
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
OUT = ROOT / "outputs" / "wave10"
SEED = 20260722
A1_CKPT = (
    ROOT
    / "outputs/wave7/wave7c/etap2/runs/W7C_A1_UNSHARED_ROLE_ENCODERS/clean/seed_20260722/best_model.pt"
)
W9_K0_CKPT = ROOT / "outputs/wave9/runs/W9_K0_MLP_CONTROL/clean/seed_20260722/best_model.pt"
# Fair R0/R1 use Wave 9 K0: same MLPPairEncoder key layout as current code,
# candidate_fp identical to A1, recorded valid within 0.002 of A1 formal freeze.
PRIMARY_CKPT = W9_K0_CKPT
PRIMARY_NAME = "W9_K0_MLP_CONTROL"
OLD_R0_CKPT = (
    ROOT / "outputs/wave10/runs/W10_R0_FROZEN_MLP_CONTROL/clean/seed_20260722/best_model.pt"
)
DEFAULT_GSN = Path.home() / "Desktop" / "GSN Graphs dysertation 2026"
PROMOTE_DELTA = 0.005


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


def _backbone_sha(state: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(state):
        if "ag_kan_residual" in k:
            continue
        h.update(k.encode())
        h.update(state[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _compose_cfg(*, residual: bool, variant: str):
    from hydra import compose, initialize_config_dir

    overrides = [
        "model=TaskA_hgt_wave7c",
        "hcr=w7c_b2_audit",
        "data.dataset.scenario=clean",
        "data.feature_ablation_profile=empirical",
        f"data.candidate_seed={SEED}",
        f"training.seed={SEED}",
        "experiment.wave=WAVE10",
        f"experiment.variant={variant}",
        "model.decoder.arch=unshared_mlp",
        "model.decoder.ablation=none",
        "model.decoder.use_triple=false",
        "model.decoder.pair_encoder.type=mlp",
        f"model.decoder.ag_kan_residual.enabled={'true' if residual else 'false'}",
        "wandb.enabled=false",
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
    if residual:
        overrides += [
            "model.decoder.ag_kan_residual.grid_size=3",
            "model.decoder.ag_kan_residual.spline_order=3",
            "model.decoder.ag_kan_residual.gate_init_logit=-3.0",
            "model.decoder.ag_kan_residual.base_scale_init=0.1",
            "model.decoder.ag_kan_residual.spline_scale_init=0.05",
            "model.decoder.ag_kan_residual.spline_l1=1.0e-5",
            "model.decoder.ag_kan_residual.train_mode=residual_only",
            "model.decoder.ag_kan_residual.type_routed=false",
            f"training.init_from_checkpoint={PRIMARY_CKPT}",
        ]
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        return compose(config_name="config", overrides=overrides)


def _load_data(cfg):
    sys.path.insert(0, str(ROOT / "src"))
    from data.PreprocessingTaskA.load_hetero_recon_data import load_recon_heterodata
    from hcr.wave7.wave7c.attach import fit_and_attach_wave7c

    train, valid, test, node_to_idx = load_recon_heterodata(cfg)
    fit_and_attach_wave7c(cfg, train, valid, test, device="cpu")
    return train, valid, test, node_to_idx


def _eval_ckpt(cfg, ckpt: Path, *, residual: bool) -> dict[str, Any]:
    import numpy as np
    import torch
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    from hcr.wave7.panel_b.subgroup_metrics import pair_bucket, safe_auprc
    from hcr.wave7.panel_b_residual.constants import N_ROLES, PAIR_B2_DIM
    from models.TaskA.checkpoint_remap import remap_wave7c_mlp_state_dict
    from train_taskA import build_model

    train, valid, test, _ = _load_data(cfg)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = build_model(cfg, train)
    with torch.no_grad():
        _ = model(train)
    state = remap_wave7c_mlp_state_dict(blob["model_state_dict"])
    missing, unexpected = model.load_state_dict(state, strict=False)
    if residual:
        bad = [k for k in missing if "ag_kan_residual" not in k]
        if bad or unexpected:
            raise RuntimeError(
                f"load mismatch missing={bad[:10]} unexpected={list(unexpected)[:10]}"
            )
    else:
        if missing or unexpected:
            raise RuntimeError(
                f"strict load failed missing={missing} unexpected={unexpected}"
            )
    model.eval()
    stats: dict[str, Any] = {
        "checkpoint": str(ckpt),
        "best_epoch_in_ckpt": int(blob.get("best_epoch", -1)),
        "candidate_seed": blob.get("candidate_seed"),
        "candidate_fingerprint": blob.get("candidate_fingerprint"),
        "backbone_sha256": _backbone_sha(blob["model_state_dict"]),
        "data_candidate_fingerprint": getattr(train, "candidate_fingerprint", None),
        "data_candidate_seed": getattr(train, "candidate_seed", None),
        "graph_fingerprint": getattr(train, "graph_fingerprint", None),
        "feature_fingerprint": getattr(train, "feature_fingerprint", None),
        "hcr_patient_train_fingerprint": getattr(train, "hcr_patient_train_fingerprint", None),
        "hcr_dim": getattr(train, "hcr_dim", None),
        "model_eval_mode": bool(not model.training),
        "n_valid": int(valid.edge_label.numel()),
        "n_test": int(test.edge_label.numel()),
    }

    def _split_metrics(data, prefix: str) -> dict[str, Any]:
        with torch.no_grad():
            logits = model(data).view(-1).cpu().numpy()
        y = data.edge_label.detach().cpu().float().view(-1).numpy()
        p = 1.0 / (1.0 + np.exp(-logits))
        out = {
            f"{prefix}_auprc": safe_auprc(y, p),
            f"{prefix}_brier": float(brier_score_loss(y, p)),
            f"{prefix}_auroc": float(roc_auc_score(y, p)) if np.unique(y).size > 1 else float("nan"),
        }
        if prefix == "valid":
            sources = list(data.candidate_source_name)
            targets = list(data.candidate_target_name)
            status = list(getattr(data, "wave7c_context_status", ["unknown"] * len(y)))
            buckets = {
                k: []
                for k in (
                    "binary_binary",
                    "count_involved",
                    "continuous_involved",
                    "mixed_type",
                )
            }
            for u, v in zip(sources, targets):
                b = pair_bucket(str(u), str(v))
                for k in buckets:
                    buckets[k].append(b[k])
            for k, flags in buckets.items():
                m = np.asarray(flags, dtype=bool)
                out[f"valid_auprc_{k}"] = safe_auprc(y[m], p[m]) if m.any() else float("nan")
                out[f"n_{k}"] = int(m.sum())
            st = np.asarray(status)
            for name, mask in (
                ("valid_context", st == "valid"),
                ("missing_context", st == "missing"),
            ):
                out[f"valid_auprc_{name}"] = (
                    float(average_precision_score(y[mask], p[mask]))
                    if mask.any() and np.unique(y[mask]).size > 1
                    else float("nan")
                )
                out[f"n_{name}"] = int(mask.sum())
        return out

    stats.update(_split_metrics(valid, "valid"))
    stats.update(_split_metrics(test, "test"))

    # Residual diagnostics on valid (if present).
    if residual and getattr(model.decoder, "ag_kan_residual", None) is not None:
        with torch.no_grad():
            _ = model(valid)
        stats["residual"] = dict(getattr(model.decoder, "last_residual_stats", {}) or {})
        # Explicit AG block probe for base/spline.
        feats = valid.hcr_features.float()
        blocks = feats.reshape(-1, N_ROLES, PAIR_B2_DIM)
        ag = blocks[:, 1, :]
        masks = getattr(valid, "hcr_role_masks", None)
        role_m = masks[:, 1] if masks is not None else None
        gated, alpha, raw = model.decoder.ag_kan_residual(ag, role_mask=role_m)
        _, base_out, spline_out = model.decoder.ag_kan_residual.kan.forward_parts(ag)
        if role_m is not None:
            m = role_m.view(-1, 1).to(base_out.dtype)
            base_out = base_out * m
            spline_out = spline_out * m
        mlp = model.decoder.role_encoders[1](ag)
        stats["residual"].update(
            {
                "gate_alpha_final": float(alpha),
                "mlp_latent_norm": float(mlp.norm(dim=-1).mean()),
                "kan_residual_norm": float(raw.norm(dim=-1).mean()),
                "ratio_residual_to_mlp": float(
                    raw.norm(dim=-1).mean() / mlp.norm(dim=-1).mean().clamp_min(1e-8)
                ),
                "base_path_norm": float(base_out.norm(dim=-1).mean()),
                "spline_path_norm": float(spline_out.norm(dim=-1).mean()),
                "ratio_spline_to_base": float(
                    spline_out.norm(dim=-1).mean()
                    / base_out.norm(dim=-1).mean().clamp_min(1e-8)
                ),
            }
        )
    return stats


def audit_baselines() -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    import torch

    rows = {}
    for name, path in (
        ("A1_formal", A1_CKPT),
        ("W9_K0", W9_K0_CKPT),
        ("W10_R0_retrained", OLD_R0_CKPT),
    ):
        if not path.is_file():
            rows[name] = {"exists": False, "path": str(path)}
            continue
        blob = torch.load(path, map_location="cpu", weights_only=False)
        fv = blob.get("final_valid_metrics") or {}
        rows[name] = {
            "exists": True,
            "path": str(path),
            "best_epoch": int(blob.get("best_epoch", -1)),
            "best_valid_metric": float(blob.get("best_valid_metric", float("nan"))),
            "final_valid_auprc": float(fv.get("auprc", float("nan"))),
            "candidate_seed": blob.get("candidate_seed"),
            "candidate_fingerprint": blob.get("candidate_fingerprint"),
            "backbone_sha256": _backbone_sha(blob["model_state_dict"]),
            "same_candidate_fp_as_A1": blob.get("candidate_fingerprint")
            == torch.load(A1_CKPT, map_location="cpu", weights_only=False).get(
                "candidate_fingerprint"
            ),
        }
    # Mark that old R0 was a retrain
    rows["verdict"] = {
        "old_W10_R0_was_retrain": rows.get("W10_R0_retrained", {}).get("backbone_sha256")
        != rows.get("A1_formal", {}).get("backbone_sha256"),
        "canonical_backbone": PRIMARY_NAME,
        "canonical_path": str(PRIMARY_CKPT),
        "a1_formal_path": str(A1_CKPT),
        "action": (
            "R0 = eval-only reload of W9 K0 (native MLPPairEncoder keys). "
            "A1 formal is remapped for audit; old W10 R0 was an invalid retrain."
        ),
    }
    return rows


def run_r1_train(epochs: int) -> Path:
    run_dir = OUT / "runs" / "W10_R1_AG_KAN_RESIDUAL_FROM_W9K0" / "clean" / f"seed_{SEED}"
    if (run_dir / "best_model.pt").exists() and (run_dir / "metrics.json").exists():
        print(f"SKIP R1 train ({run_dir})", flush=True)
        return run_dir / "best_model.pt"
    run_dir.mkdir(parents=True, exist_ok=True)
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
        f"training.init_from_checkpoint={PRIMARY_CKPT}",
        "experiment.wave=WAVE10",
        "experiment.variant=W10_R1_AG_KAN_RESIDUAL_FROM_W9K0",
        "experiment.intervention=W10_R1_AG_KAN_RESIDUAL_FROM_W9K0",
        "model.decoder.arch=unshared_mlp",
        "model.decoder.ablation=none",
        "model.decoder.use_triple=false",
        "model.decoder.pair_encoder.type=mlp",
        "model.decoder.ag_kan_residual.enabled=true",
        "model.decoder.ag_kan_residual.input_dim=40",
        "model.decoder.ag_kan_residual.output_dim=8",
        "model.decoder.ag_kan_residual.spline_order=3",
        "model.decoder.ag_kan_residual.grid_size=3",
        "model.decoder.ag_kan_residual.grid_range=[-3.0,3.0]",
        "model.decoder.ag_kan_residual.grid_update=false",
        "model.decoder.ag_kan_residual.base_activation=silu",
        "model.decoder.ag_kan_residual.base_scale_init=0.1",
        "model.decoder.ag_kan_residual.spline_scale_init=0.05",
        "model.decoder.ag_kan_residual.gate_init_logit=-3.0",
        "model.decoder.ag_kan_residual.spline_l1=1.0e-5",
        "model.decoder.ag_kan_residual.type_routed=false",
        "model.decoder.ag_kan_residual.train_mode=residual_only",
        "wandb.enabled=true",
        "wandb.group=TaskA_WAVE10_KAN_RESIDUAL",
        "wandb.job_type=wave10_fair",
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
    print("\nRUN R1:", " ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), env=_env())
    if proc.returncode != 0:
        raise SystemExit(f"R1 train failed rc={proc.returncode}")
    print(f"R1 train done in {time.time()-t0:.1f}s", flush=True)
    return run_dir / "best_model.pt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--skip-r1-train", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audits").mkdir(exist_ok=True)
    (OUT / "summaries").mkdir(exist_ok=True)

    print("=== Audit frozen baselines ===", flush=True)
    audit = audit_baselines()
    (OUT / "audits" / "W10_R0_BASELINE_AUDIT.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit["verdict"], indent=2), flush=True)

    print("\n=== R0 eval-only (W9 K0 primary) ===", flush=True)
    cfg_r0 = _compose_cfg(residual=False, variant="W10_R0_EVAL_W9K0")
    r0 = _eval_ckpt(cfg_r0, PRIMARY_CKPT, residual=False)
    r0_dir = OUT / "runs" / "W10_R0_EVAL_W9K0" / "clean" / f"seed_{SEED}"
    r0_dir.mkdir(parents=True, exist_ok=True)
    (r0_dir / "metrics.json").write_text(json.dumps(r0, indent=2, default=float))

    # A1 remapped reload (formal freeze) for audit transparency.
    if A1_CKPT.is_file():
        cfg_a1 = _compose_cfg(residual=False, variant="W10_R0_EVAL_A1_REMAPPED")
        a1 = _eval_ckpt(cfg_a1, A1_CKPT, residual=False)
        (OUT / "audits" / "W10_A1_REMAPPED_RELOAD.json").write_text(
            json.dumps(a1, indent=2, default=float)
        )

    if not args.skip_r1_train:
        r1_ckpt = run_r1_train(args.epochs)
    else:
        r1_ckpt = OUT / "runs" / "W10_R1_AG_KAN_RESIDUAL_FROM_W9K0" / "clean" / f"seed_{SEED}" / "best_model.pt"
        if not r1_ckpt.is_file():
            raise SystemExit("R1 checkpoint missing; drop --skip-r1-train")

    print("\n=== R1 eval (from W9 K0 + residual) ===", flush=True)
    cfg_r1 = _compose_cfg(residual=True, variant="W10_R1_AG_KAN_RESIDUAL_FROM_W9K0")
    r1 = _eval_ckpt(cfg_r1, r1_ckpt, residual=True)
    (r1_ckpt.parent / "metrics_fair.json").write_text(json.dumps(r1, indent=2, default=float))

    delta = float(r1["valid_auprc"]) - float(r0["valid_auprc"])
    if delta >= PROMOTE_DELTA:
        action = "promote_multiseed"
    elif delta < -PROMOTE_DELTA:
        action = "stop"
    else:
        action = "diagnostic_only"

    residual = r1.get("residual") or {}
    decision = {
        "protocol": "fair_W9K0_eval_only_R0",
        "canonical_backbone": PRIMARY_NAME,
        "canonical_path": str(PRIMARY_CKPT),
        "selection_metric": "valid_auprc",
        "R0_mode": "eval_only_no_retrain",
        "R0_valid": r0["valid_auprc"],
        "R1_valid": r1["valid_auprc"],
        "R1_minus_R0": delta,
        "R0_test": r0["test_auprc"],
        "R1_test": r1["test_auprc"],
        "R0_brier": r0["valid_brier"],
        "R1_brier": r1["valid_brier"],
        "fingerprints": {
            "candidate": r0.get("candidate_fingerprint"),
            "data_candidate": r0.get("data_candidate_fingerprint"),
            "graph": r0.get("graph_fingerprint"),
            "feature": r0.get("feature_fingerprint"),
            "hcr_patient_train": r0.get("hcr_patient_train_fingerprint"),
            "backbone_sha_R0": r0.get("backbone_sha256"),
            "backbone_sha_R1": r1.get("backbone_sha256"),
            "backbone_match": r0.get("backbone_sha256") == r1.get("backbone_sha256"),
        },
        "residual_diagnostics": {
            "alpha_final": residual.get("gate_alpha_final", residual.get("gate_alpha")),
            "ratio_residual_to_mlp": residual.get("ratio_residual_to_mlp"),
            "base_path_norm": residual.get("base_path_norm"),
            "spline_path_norm": residual.get("spline_path_norm"),
            "ratio_spline_to_base": residual.get("ratio_spline_to_base"),
            "mlp_latent_norm": residual.get("mlp_latent_norm"),
            "kan_residual_norm": residual.get("kan_residual_norm"),
        },
        "pair_types_R0": {k: r0[k] for k in r0 if k.startswith("valid_auprc_")},
        "pair_types_R1": {k: r1[k] for k in r1 if k.startswith("valid_auprc_")},
        "action": action,
        "promote_to_multiseed": action == "promote_multiseed",
        "note": (
            "Do not promote on previous retrained R0. "
            "α≈0 → KAN unused; small spline/large base → mostly SiLU; "
            "gain only on count/continuous → consider T1 once."
        ),
    }
    (OUT / "summaries" / "WAVE10_FAIR_DECISION.json").write_text(
        json.dumps(decision, indent=2, default=float)
    )
    (OUT / "WAVE10_FAIR_DECISION.json").write_text(json.dumps(decision, indent=2, default=float))

    csv_path = OUT / "summaries" / "wave10_fair_screen.csv"
    with csv_path.open("w", newline="") as f:
        fields = [
            "variant",
            "mode",
            "valid_auprc",
            "test_auprc",
            "valid_brier",
            "backbone_sha256",
            "best_epoch_in_ckpt",
            "alpha_final",
            "ratio_residual_to_mlp",
            "base_path_norm",
            "spline_path_norm",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow(
            {
                "variant": "W10_R0_EVAL_W9K0",
                "mode": "eval_only",
                "valid_auprc": r0["valid_auprc"],
                "test_auprc": r0["test_auprc"],
                "valid_brier": r0["valid_brier"],
                "backbone_sha256": r0["backbone_sha256"],
                "best_epoch_in_ckpt": r0["best_epoch_in_ckpt"],
                "alpha_final": "",
                "ratio_residual_to_mlp": "",
                "base_path_norm": "",
                "spline_path_norm": "",
            }
        )
        w.writerow(
            {
                "variant": "W10_R1_AG_KAN_RESIDUAL_FROM_W9K0",
                "mode": "residual_from_W9K0",
                "valid_auprc": r1["valid_auprc"],
                "test_auprc": r1["test_auprc"],
                "valid_brier": r1["valid_brier"],
                "backbone_sha256": r1["backbone_sha256"],
                "best_epoch_in_ckpt": r1["best_epoch_in_ckpt"],
                "alpha_final": residual.get("gate_alpha_final", residual.get("gate_alpha")),
                "ratio_residual_to_mlp": residual.get("ratio_residual_to_mlp"),
                "base_path_norm": residual.get("base_path_norm"),
                "spline_path_norm": residual.get("spline_path_norm"),
            }
        )

    print(json.dumps(decision, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
