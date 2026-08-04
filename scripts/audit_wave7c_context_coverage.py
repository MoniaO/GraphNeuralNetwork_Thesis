#!/usr/bin/env python3
"""Build CONTEXT_COVERAGE_R1.csv + zero-input encoder probe for Wave 7C."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "wave7" / "panel_b_r1_audit"
SEED = 20260722
R1_CKPT = (
    ROOT
    / "outputs/wave7/panel_b_residual/runs/W7BR_R1_B2_SHARED_PAIR_ENCODER/clean"
    / f"seed_{SEED}/best_model.pt"
)


def main() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from data.PreprocessingTaskA.load_hetero_recon_data import load_recon_heterodata
    from hcr.variable_spec import VariableType
    from hcr.variable_specs_v3 import VARIABLE_SPECS
    from hcr.wave7.wave7c.attach import fit_and_attach_wave7c
    from models.TaskA.wave7c_decoder import Wave7CDecoder

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
                "experiment.variant=W7C_A0_R1_SHARED_PAIR_ENCODER",
                "wandb.enabled=false",
            ],
        )

    train, valid, test, _ = load_recon_heterodata(cfg)
    fit_and_attach_wave7c(cfg, train, valid, test, device="cpu")

    def kind(name: str) -> str:
        spec = VARIABLE_SPECS.get(str(name))
        if spec is None:
            return "unknown"
        if spec.variable_type == VariableType.BINARY:
            return "binary"
        if spec.variable_type == VariableType.COUNT:
            return "count"
        return "continuous"

    dec = Wave7CDecoder(hidden_channels=32, arch="shared_mlp", ablation="none")
    dec.eval()
    probe: dict = {}
    with torch.no_grad():
        zero_in = torch.zeros(1, 40)
        zero_lat_init = dec.pair_encoder(zero_in).cpu().numpy().reshape(-1)
    probe["init"] = {
        "zero_input_shape": [1, 40],
        "zero_latent": zero_lat_init.tolist(),
        "zero_latent_norm": float(np.linalg.norm(zero_lat_init)),
        "note": "Non-zero norm ⇒ Linear/LayerNorm biases encode missingness.",
    }
    if R1_CKPT.exists():
        blob = torch.load(R1_CKPT, map_location="cpu", weights_only=False)
        mapped = {
            k.replace("decoder.b2_encoder.", ""): v
            for k, v in blob["model_state_dict"].items()
            if k.startswith("decoder.b2_encoder.")
        }
        dec.pair_encoder.load_state_dict(mapped)
        with torch.no_grad():
            zero_lat_r1 = dec.pair_encoder(zero_in).cpu().numpy().reshape(-1)
        probe["trained_r1"] = {
            "checkpoint": str(R1_CKPT.relative_to(ROOT)),
            "zero_latent": zero_lat_r1.tolist(),
            "zero_latent_norm": float(np.linalg.norm(zero_lat_r1)),
        }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ZERO_INPUT_PROBE.json").write_text(json.dumps(probe, indent=2))

    rows = []
    for data, split in ((train, "train"), (valid, "valid"), (test, "test")):
        n = int(data.hcr_features.size(0))
        feats = data.hcr_features.detach().cpu().numpy()
        sources = list(data.candidate_source_name)
        targets = list(data.candidate_target_name)
        y = np.asarray(data.wave7c_edge_label_audit, dtype=float)
        status = list(data.wave7c_context_status)
        zs = list(data.wave7c_selected_z)
        reasons = list(data.wave7c_context_reason)
        sources_sel = list(data.wave7c_context_source)
        with torch.no_grad():
            lat = dec.encode_pairs(torch.as_tensor(feats)).cpu().numpy()

        for i in range(n):
            a, g = str(sources[i]), str(targets[i])
            z = str(zs[i])
            blocks = feats[i].reshape(3, 40)
            raw_norms = [float(np.linalg.norm(blocks[r])) for r in range(3)]
            lat_norms = [float(np.linalg.norm(lat[i, r])) for r in range(3)]
            supports = [float(blocks[r, 32]) for r in range(3)]
            supported = [bool(blocks[r, 33] > 0.5) for r in range(3)]
            types = {kind(a), kind(g), kind(z) if z else "none"}
            if types <= {"binary", "none"}:
                cand_type = "binary_motif" if z else "binary_pair"
            elif "count" in types and "continuous" in types:
                cand_type = "mixed_count_continuous"
            elif "count" in types:
                cand_type = "count_involved"
            elif "continuous" in types:
                cand_type = "continuous_involved"
            else:
                cand_type = "other"

            rows.append(
                {
                    "candidate_id": f"{split}:{i}:{a}->{g}",
                    "split": split,
                    "edge_label": float(y[i]) if np.isfinite(y[i]) else np.nan,
                    "A": a,
                    "G": g,
                    "Z": z,
                    "context_status": status[i],
                    "context_selection_reason": reasons[i],
                    "context_source": sources_sel[i],
                    "A_type": kind(a),
                    "Z_type": kind(z) if z else "",
                    "G_type": kind(g),
                    "AZ_supported": supported[0],
                    "AG_supported": supported[1],
                    "ZG_supported": supported[2],
                    "AZ_support": supports[0],
                    "AG_support": supports[1],
                    "ZG_support": supports[2],
                    "AZ_raw_norm": raw_norms[0],
                    "AG_raw_norm": raw_norms[1],
                    "ZG_raw_norm": raw_norms[2],
                    "AZ_latent_norm": lat_norms[0],
                    "AG_latent_norm": lat_norms[1],
                    "ZG_latent_norm": lat_norms[2],
                    "candidate_was_retained": True,
                    "candidate_type": cand_type,
                }
            )

    df = pd.DataFrame(rows)
    export_cols = [
        "candidate_id",
        "split",
        "edge_label",
        "A",
        "G",
        "Z",
        "context_status",
        "context_selection_reason",
        "context_source",
        "A_type",
        "Z_type",
        "G_type",
        "AZ_supported",
        "AG_supported",
        "ZG_supported",
        "AZ_support",
        "AG_support",
        "ZG_support",
        "AZ_raw_norm",
        "AG_raw_norm",
        "ZG_raw_norm",
        "AZ_latent_norm",
        "AG_latent_norm",
        "ZG_latent_norm",
        "candidate_was_retained",
    ]
    csv_path = OUT / "CONTEXT_COVERAGE_R1.csv"
    df[export_cols].to_csv(csv_path, index=False)

    summary: dict = {"n_candidates": int(len(df)), "zero_input_probe": probe}
    vc = df["context_status"].value_counts(dropna=False)
    summary["context_status_counts"] = {str(k): int(v) for k, v in vc.items()}
    summary["context_status_fraction"] = {
        str(k): float(v / len(df)) for k, v in vc.items()
    }

    ct = pd.crosstab(df["context_status"], df["edge_label"], dropna=False)
    summary["context_status_x_edge_label"] = {
        str(i): {str(c): int(ct.loc[i, c]) for c in ct.columns} for i in ct.index
    }

    prev = {}
    for st, g in df.groupby("context_status"):
        yv = g["edge_label"].to_numpy()
        yv = yv[np.isfinite(yv)]
        prev[str(st)] = {
            "n": int(len(g)),
            "n_labeled": int(len(yv)),
            "positive_prevalence": float(yv.mean()) if len(yv) else float("nan"),
        }
    summary["prevalence_by_context_status"] = prev

    by_split = df.groupby(["split", "context_status"]).size().unstack(fill_value=0)
    summary["context_status_by_split"] = {
        str(idx): {str(c): int(by_split.loc[idx, c]) for c in by_split.columns}
        for idx in by_split.index
    }
    by_type = df.groupby(["candidate_type", "context_status"]).size().unstack(fill_value=0)
    summary["context_status_by_candidate_type"] = {
        str(idx): {str(c): int(by_type.loc[idx, c]) for c in by_type.columns}
        for idx in by_type.index
    }

    def _prev(status: str) -> float:
        yv = df.loc[df["context_status"] == status, "edge_label"].to_numpy()
        yv = yv[np.isfinite(yv)]
        return float(yv.mean()) if len(yv) else float("nan")

    summary["P_y1_given_Z_valid"] = _prev("valid")
    summary["P_y1_given_Z_missing"] = _prev("missing")
    summary["prevalence_gap_valid_minus_missing"] = (
        summary["P_y1_given_Z_valid"] - summary["P_y1_given_Z_missing"]
    )
    summary["pair_support_by_role"] = {
        role: {
            "mean_support": float(df[f"{role}_support"].mean()),
            "frac_supported": float(df[f"{role}_supported"].astype(float).mean()),
            "mean_raw_norm": float(df[f"{role}_raw_norm"].mean()),
            "mean_latent_norm": float(df[f"{role}_latent_norm"].mean()),
        }
        for role in ("AZ", "AG", "ZG")
    }
    miss = df[df["context_status"] == "missing"]
    if len(miss):
        summary["missing_context_zero_block"] = {
            "AZ_raw_norm_mean": float(miss["AZ_raw_norm"].mean()),
            "ZG_raw_norm_mean": float(miss["ZG_raw_norm"].mean()),
            "AZ_latent_norm_mean": float(miss["AZ_latent_norm"].mean()),
            "ZG_latent_norm_mean": float(miss["ZG_latent_norm"].mean()),
        }
    znorm = probe.get("trained_r1", probe["init"])["zero_latent_norm"]
    summary["zero_block_latent_nonzero"] = bool(znorm > 1e-8)

    (OUT / "CONTEXT_COVERAGE_R1_SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {csv_path}")
    print(f"Wrote {OUT / 'ZERO_INPUT_PROBE.json'}")


if __name__ == "__main__":
    main()
