#!/usr/bin/env python3
"""Wave 10A — diagnose Wave 9 K1 KAN failure (no new training).

Writes:
  outputs/wave10/audits/W10_KAN_FAILURE_DIAGNOSTIC.json
  outputs/wave10/audits/W10_KAN_INPUT_GRID_AUDIT.csv
  outputs/wave10/audits/W10_KAN_PAIR_TYPE_METRICS.csv
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "outputs" / "wave10" / "audits"
K1_DIR = ROOT / "outputs" / "wave9" / "runs" / "W9_K1_PAIR_KAN_FIXED" / "clean" / "seed_20260722"
K0_DIR = ROOT / "outputs" / "wave9" / "runs" / "W9_K0_MLP_CONTROL" / "clean" / "seed_20260722"
SEED = 20260722
ROLES = ("AZ", "AG", "ZG")
DEFAULT_GSN = Path.home() / "Desktop" / "GSN Graphs dysertation 2026"


def _env_bootstrap() -> None:
    os.environ.setdefault("GSN_PROJECT_ROOT", str(DEFAULT_GSN))
    os.environ.setdefault(
        "PHARMA_DATA_ROOT",
        str(Path(os.environ["GSN_PROJECT_ROOT"]) / "2 v3. Data" / "dataset_v3"),
    )
    os.environ.setdefault("PYTHONPATH", f"{ROOT / 'src'}:{ROOT}")


def _kan_layer_parts(layer, x: torch.Tensor) -> dict[str, float]:
    x2 = x.reshape(-1, layer.in_features)
    with torch.no_grad():
        base = F.linear(layer.base_activation(x2), layer.base_weight)
        spline = F.linear(
            layer.b_splines(x2).view(x2.size(0), -1),
            layer.scaled_spline_weight.view(layer.out_features, -1),
        )
        if layer.bias is not None:
            out = base + spline + layer.bias
        else:
            out = base + spline
        lo, hi = layer.grid_range
        oog = float(((x2 < lo) | (x2 > hi)).float().mean())
    return {
        "base_weight_norm": float(layer.base_weight.detach().norm()),
        "spline_coefficient_norm": float(layer.spline_weight.detach().norm()),
        "max_abs_spline_coefficient": float(layer.spline_weight.detach().abs().max()),
        "fraction_zero_spline_coefficients": float(
            (layer.spline_weight.detach().abs() < 1e-12).float().mean()
        ),
        "base_output_norm": float(base.norm(dim=-1).mean()),
        "spline_output_norm": float(spline.norm(dim=-1).mean()),
        "total_output_norm": float(out.norm(dim=-1).mean()),
        "ratio_spline_to_base_output": float(
            spline.norm(dim=-1).mean() / base.norm(dim=-1).mean().clamp_min(1e-8)
        ),
        "input_out_of_grid_fraction": oog,
    }


def main() -> None:
    _env_bootstrap()
    OUT.mkdir(parents=True, exist_ok=True)

    ckpt_path = K1_DIR / "best_model.pt"
    metrics_k1 = json.loads((K1_DIR / "metrics.json").read_text())
    metrics_k0 = json.loads((K0_DIR / "metrics.json").read_text())
    if not ckpt_path.exists():
        raise SystemExit(f"missing K1 checkpoint: {ckpt_path}")

    from hydra import compose, initialize_config_dir
    from sklearn.metrics import average_precision_score, brier_score_loss

    from data.PreprocessingTaskA.load_hetero_recon_data import load_recon_heterodata
    from hcr.wave7.panel_b.subgroup_metrics import pair_bucket, safe_auprc
    from hcr.wave7.panel_b_residual.constants import N_ROLES, PAIR_B2_DIM
    from hcr.wave7.wave7c.attach import fit_and_attach_wave7c
    from models.TaskA.pair_encoders.kan import KANPairEncoder
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
                "experiment.wave=WAVE9",
                "experiment.variant=W9_K1_PAIR_KAN_FIXED",
                "model.decoder.arch=unshared_mlp",
                "model.decoder.ablation=none",
                "model.decoder.use_triple=false",
                "model.decoder.pair_encoder.type=kan",
                "model.decoder.pair_encoder.grid_size=5",
                "model.decoder.pair_encoder.spline_order=3",
                "model.decoder.pair_encoder.grid_range=[-3.0,3.0]",
                "model.decoder.pair_encoder.grid_update=false",
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
            ],
        )

    train, valid, test, _ = load_recon_heterodata(cfg)
    fit_and_attach_wave7c(cfg, train, valid, test, device="cpu")
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(cfg, train)
    with torch.no_grad():
        _ = model(train)
    model.load_state_dict(blob["model_state_dict"])
    model.eval()

    # ---- role / pair-type / context metrics on valid ----
    with torch.no_grad():
        logits = model(valid).view(-1).cpu().numpy()
    y = valid.edge_label.detach().cpu().float().view(-1).numpy()
    p = 1.0 / (1.0 + np.exp(-logits))
    sources = list(valid.candidate_source_name)
    targets = list(valid.candidate_target_name)
    status = list(getattr(valid, "wave7c_context_status", ["unknown"] * len(y)))

    pair_rows = []
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
        pair_rows.append(
            {
                "subset": k,
                "n": int(m.sum()),
                "valid_auprc": safe_auprc(y[m], p[m]) if m.any() else float("nan"),
                "valid_brier": float(brier_score_loss(y[m], p[m]))
                if m.any() and np.unique(y[m]).size > 1
                else float("nan"),
            }
        )
    st = np.asarray(status)
    for name, mask in (
        ("valid_context", st == "valid"),
        ("missing_context", st == "missing"),
        ("matched_shuffled_context", st == "matched_shuffled"),
    ):
        pair_rows.append(
            {
                "subset": name,
                "n": int(mask.sum()),
                "valid_auprc": float(average_precision_score(y[mask], p[mask]))
                if mask.any() and np.unique(y[mask]).size > 1
                else float("nan"),
                "valid_brier": float(brier_score_loss(y[mask], p[mask]))
                if mask.any()
                else float("nan"),
            }
        )
    pair_rows.append(
        {
            "subset": "all_valid",
            "n": int(len(y)),
            "valid_auprc": safe_auprc(y, p),
            "valid_brier": float(brier_score_loss(y, p)),
        }
    )
    with (OUT / "W10_KAN_PAIR_TYPE_METRICS.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subset", "n", "valid_auprc", "valid_brier"])
        w.writeheader()
        w.writerows(pair_rows)

    # ---- per-role KAN diagnostics on train AG/AZ/ZG blocks ----
    feats = train.hcr_features.detach().cpu().float()
    batch = feats.size(0)
    blocks = feats.reshape(batch, N_ROLES, PAIR_B2_DIM)
    role_encoders = model.decoder.role_encoders
    assert role_encoders is not None

    role_diag: dict[str, Any] = {}
    grid_rows = []
    for r, name in enumerate(ROLES):
        enc = role_encoders[r]
        assert isinstance(enc, KANPairEncoder)
        x = blocks[:, r, :]
        # Probe with requires_grad for gradient norms on a small batch.
        x_g = x[: min(256, batch)].clone().requires_grad_(True)
        enc.zero_grad(set_to_none=True)
        out = enc(x_g)
        loss = out.pow(2).mean()
        loss.backward()
        g_base = 0.0
        g_spline = 0.0
        for layer in (enc.layer1, enc.layer2):
            if layer.base_weight.grad is not None:
                g_base += float(layer.base_weight.grad.norm())
            if layer.spline_weight.grad is not None:
                g_spline += float(layer.spline_weight.grad.norm())
        l1 = _kan_layer_parts(enc.layer1, x)
        l2 = _kan_layer_parts(enc.layer2, enc.norm(enc.layer1(x)))
        role_diag[name] = {
            "layer1": l1,
            "layer2": l2,
            "gradient_norm_base": g_base,
            "gradient_norm_spline": g_spline,
            "output_norm": float(out.detach().norm(dim=-1).mean()),
        }
        for slot in range(PAIR_B2_DIM):
            col = x[:, slot].numpy()
            lo, hi = -3.0, 3.0
            grid_rows.append(
                {
                    "role": name,
                    "slot": slot,
                    "min": float(col.min()),
                    "p05": float(np.percentile(col, 5)),
                    "median": float(np.median(col)),
                    "p95": float(np.percentile(col, 95)),
                    "max": float(col.max()),
                    "fraction_outside_-3_3": float(((col < lo) | (col > hi)).mean()),
                    "fraction_zero": float((np.abs(col) < 1e-12).mean()),
                }
            )

    with (OUT / "W10_KAN_INPUT_GRID_AUDIT.csv").open("w", newline="") as f:
        fields = list(grid_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(grid_rows)

    oog = [r["fraction_outside_-3_3"] for r in grid_rows]
    adaptive_justified = bool(np.mean(oog) > 0.05 or np.max(oog) > 0.20)

    # Role contribution proxy: zero each role's 40D block on a feature clone.
    role_ablation = {}
    valid_feats_orig = valid.hcr_features.clone()
    with torch.no_grad():
        base_auprc = safe_auprc(y, p)
        for r, name in enumerate(ROLES):
            vf = valid_feats_orig.clone()
            vb = vf.reshape(-1, N_ROLES, PAIR_B2_DIM)
            vb[:, r, :] = 0.0
            valid.hcr_features = vf
            logits_r = model(valid).view(-1).cpu().numpy()
            pr = 1.0 / (1.0 + np.exp(-logits_r))
            role_ablation[name] = {
                "valid_auprc_zeroed_role_input": safe_auprc(y, pr),
                "delta_vs_full": float(safe_auprc(y, pr) - base_auprc),
            }
        valid.hcr_features = valid_feats_orig

    diagnostic = {
        "source_checkpoint": str(ckpt_path),
        "best_epoch": int(blob.get("best_epoch", -1)),
        "k0_metrics": metrics_k0,
        "k1_metrics": metrics_k1,
        "delta_valid_k1_minus_k0": float(metrics_k1["valid_auprc"])
        - float(metrics_k0["valid_auprc"]),
        "role_kan_diagnostics": role_diag,
        "role_input_zero_ablation": role_ablation,
        "pair_type_summary": {r["subset"]: r for r in pair_rows},
        "grid_audit_summary": {
            "mean_fraction_outside_grid": float(np.mean(oog)),
            "max_fraction_outside_grid": float(np.max(oog)),
            "adaptive_grid_justified": adaptive_justified,
            "note": "Adaptive G1 only if mean>5% or max>20% OOG on train role slots.",
        },
        "recommendations": {
            "prefer_ag_residual": True,
            "reason": (
                "Wave 9 full replacement lost −3.5pp valid; C1/C2 show AG carries "
                "the motif signal. Start with near-off AG KAN residual (α≈0.047)."
            ),
            "run_adaptive_grid": adaptive_justified,
            "type_routed_candidate": bool(
                (pair_rows[1]["valid_auprc"] or 0) < (pair_rows[0]["valid_auprc"] or 1)
                or (pair_rows[2]["valid_auprc"] or 0) < (pair_rows[0]["valid_auprc"] or 1)
            ),
        },
    }
    (OUT / "W10_KAN_FAILURE_DIAGNOSTIC.json").write_text(
        json.dumps(diagnostic, indent=2, default=float)
    )
    print(json.dumps({
        "wrote": [
            str(OUT / "W10_KAN_FAILURE_DIAGNOSTIC.json"),
            str(OUT / "W10_KAN_INPUT_GRID_AUDIT.csv"),
            str(OUT / "W10_KAN_PAIR_TYPE_METRICS.csv"),
        ],
        "delta_valid": diagnostic["delta_valid_k1_minus_k0"],
        "adaptive_grid_justified": adaptive_justified,
        "ag_zero_delta": role_ablation.get("AG"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
