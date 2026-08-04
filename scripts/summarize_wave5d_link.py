#!/usr/bin/env python3
"""Summarize Wave 5D link-prediction results and write Task A decision."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/link_prediction/wave5d_clean.yaml",
    )
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    pred_dir = ROOT / cfg["paths"]["output_dir"] / "predictions"
    out_dir = ROOT / cfg["paths"]["output_dir"] / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_files = sorted(pred_dir.glob("*_metrics.csv"))
    if not metric_files:
        raise SystemExit(f"No metrics under {pred_dir}")

    metrics = pd.concat([pd.read_csv(p) for p in metric_files], ignore_index=True)
    metrics.to_csv(out_dir / "per_seed_metrics.csv", index=False)

    test = metrics[metrics["split"] == "test"].copy()
    summary = (
        test.groupby("variant", as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            auprc_mean=("auprc", "mean"),
            auprc_std=("auprc", "std"),
            auroc_mean=("auroc", "mean"),
            brier_mean=("brier", "mean"),
            mrr_mean=("mrr", "mean"),
        )
        .sort_values("auprc_mean", ascending=False)
    )
    summary.to_csv(out_dir / "variant_summary.csv", index=False)

    def mean_auprc(v: str) -> float:
        sub = test[test["variant"] == v]["auprc"]
        return float(sub.mean()) if len(sub) else float("nan")

    def seeds_better(ref: str, other: str) -> int:
        a = test[test["variant"] == ref].set_index("seed")["auprc"]
        b = test[test["variant"] == other].set_index("seed")["auprc"]
        common = a.index.intersection(b.index)
        return int((a.loc[common] > b.loc[common]).sum())

    l4 = mean_auprc("L4_final")
    l2 = mean_auprc("L2_structural_hcr")
    checks = {
        "L4_gt_L2_mean": bool(l4 > l2) if np.isfinite(l4) and np.isfinite(l2) else False,
        "L4_gt_L2_seeds": seeds_better("L4_final", "L2_structural_hcr"),
        "L4_gt_L5": bool(l4 > mean_auprc("L5_patient_shuffle")),
        "L4_gt_L6": bool(l4 > mean_auprc("L6_context_shuffle")),
        "L4_gt_L7": bool(l4 > mean_auprc("L7_matched_random_context")),
        "L4_gt_L8": bool(l4 > mean_auprc("L8_random_path_weights")),
        "brier_ok": bool(
            mean_auprc("L4_final") == mean_auprc("L4_final")  # placeholder overwritten
        ),
    }
    brier_l4 = float(test[test.variant == "L4_final"]["brier"].mean())
    brier_l2 = float(test[test.variant == "L2_structural_hcr"]["brier"].mean())
    checks["brier_ok"] = bool(brier_l4 <= brier_l2 + 0.01)

    paired = []
    for other in [
        "L2_structural_hcr",
        "L5_patient_shuffle",
        "L6_context_shuffle",
        "L7_matched_random_context",
        "L8_random_path_weights",
        "L0_hgt",
        "L3_path_support",
    ]:
        paired.append(
            {
                "reference": "L4_final",
                "variant": other,
                "delta_auprc_mean": l4 - mean_auprc(other),
                "n_seeds_ref_better": seeds_better("L4_final", other),
            }
        )
    pd.DataFrame(paired).to_csv(out_dir / "paired_deltas.csv", index=False)

    # Role FPR aggregate
    role_files = sorted(pred_dir.glob("*_test_role_fpr.csv"))
    if role_files:
        roles = []
        for p in role_files:
            df = pd.read_csv(p)
            # parse variant/seed from name
            stem = p.name.replace("_test_role_fpr.csv", "")
            # {variant}_seed{seed}
            if "_seed" in stem:
                variant, seed_s = stem.rsplit("_seed", 1)
                df["variant"] = variant
                df["seed"] = int(seed_s)
            roles.append(df)
        role_all = pd.concat(roles, ignore_index=True)
        role_all.to_csv(out_dir / "role_specific_fpr.csv", index=False)

    # Final predicted edges from L4
    final_rows = []
    for seed in cfg["seeds"]:
        p = pred_dir / f"L4_final_seed{seed}_test.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df["rank_global"] = df["probability"].rank(ascending=False, method="first")
        df["rank_within_edge_type"] = df.groupby("edge_type")["probability"].rank(
            ascending=False, method="first"
        )
        df["p_final"] = df["probability"]
        df["calibrated_probability"] = df["probability"]
        final_rows.append(df)
    if final_rows:
        final = pd.concat(final_rows, ignore_index=True)
        final.to_csv(out_dir / "final_predicted_edges.csv", index=False)

    # Path recovery placeholder (filled if path metrics exported later)
    pd.DataFrame(
        [
            {
                "variant": "L4_final",
                "true_path_coverage": None,
                "note": "path recovery computed in Wave 5C/5E bridge; edge AUPRC is primary here",
            }
        ]
    ).to_csv(out_dir / "path_recovery_summary.csv", index=False)

    path_ok = None  # optional
    edge_pass = (
        checks["L4_gt_L2_mean"]
        and checks["L4_gt_L2_seeds"] >= 4
        and checks["L4_gt_L5"]
        and checks["L4_gt_L6"]
        and checks["L4_gt_L7"]
        and checks["L4_gt_L8"]
        and checks["brier_ok"]
    )
    if edge_pass:
        result_label = "PASS_L4_FINAL"
        final_edge_predictor = "L4_final"
        note = "Patient-conditioned path support improves edge AUPRC over L2."
    elif np.isfinite(l4) and np.isfinite(l2) and l4 <= l2:
        result_label = "PASS_L2_EDGE_WNERW_RERANKER"
        final_edge_predictor = "L2_structural_hcr"
        note = (
            "Path support did not improve single-edge AUPRC; "
            "freeze L2 as edge predictor and keep WNERW as path reranker."
        )
    else:
        result_label = "FAIL"
        final_edge_predictor = "L2_structural_hcr"
        note = "Wave 5D success criteria not met."

    decision = {
        "wave": "WAVE5D",
        "task": "TaskA_link_prediction",
        "scenario": cfg["data"]["scenario"],
        "primary_metric": "auprc",
        "auprc_l4": l4,
        "auprc_l2": l2,
        "delta_l4_minus_l2": (l4 - l2) if np.isfinite(l4) and np.isfinite(l2) else None,
        "checks": checks,
        "final_edge_predictor": final_edge_predictor,
        "result_label": result_label,
        "note": note,
        "path_coverage_check": path_ok,
        "frozen_path_energy": {
            "temperature": 0.5,
            "length_penalty": 0.1,
            "gate_weight": 2.0,
            "node_activity_weight": 0.0,
            "static_hcr_weight": 0.0,
        },
    }
    (out_dir / "TASK_A_LINK_PREDICTION_DECISION.json").write_text(
        json.dumps(decision, indent=2)
    )

    print("=== VARIANT SUMMARY (test) ===")
    print(summary.to_string(index=False))
    print("=== DECISION ===")
    print(json.dumps(decision, indent=2))

    # Upload aggregate tables to W&B
    try:
        from link_prediction.wave5d.wandb_logging import log_summary_tables

        url = log_summary_tables(cfg, out_dir)
        if url:
            print(f"W&B summary run: {url}")
    except Exception as exc:  # noqa: BLE001
        print(f"W&B summary upload failed: {exc}")


if __name__ == "__main__":
    main()
