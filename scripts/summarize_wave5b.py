#!/usr/bin/env python3
"""Wave 5B — final paired deltas + WAVE5B_DECISION.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wnerw.wave5b.paired_summary import (
    decide_wave5c,
    paired_deltas,
    per_seed_wins,
    summarize_panel_e,
    summarize_panel_t,
    write_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/wnerw/wave5b.yaml",
    )
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    base = ROOT / cfg["paths"]["output_dir"]
    summary_dir = base / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    panel_t = pd.read_csv(base / "panel_t" / "query_metrics.csv")
    panel_t_sum = summarize_panel_t(panel_t)
    panel_t_sum.to_csv(base / "panel_t" / "summary.csv", index=False)

    test_path = base / "panel_e" / "query_metrics_test.csv"
    if not test_path.exists():
        raise SystemExit(f"Missing {test_path}; run run_wave5b_energy.py first")
    panel_e = pd.read_csv(test_path)
    panel_e_sum = summarize_panel_e(panel_e)
    panel_e_sum.to_csv(base / "panel_e" / "summary_test.csv", index=False)

    selected = json.loads(
        (base / "tuning" / "selected_energy_config.json").read_text()
    )

    comparisons = [
        ("E2_hcr2", "E1_hgt"),
        ("E2_hcr2", "E4_patient_shuffle"),
        ("E2_hcr2", "E5_edgewise_shuffle"),
        ("E3_hcr2_uncertainty", "E2_hcr2"),
    ]
    delta_frames = []
    for metric in [
        "reciprocal_rank_at_100",
        "true_path_mass",
        "true_path_hit_at_5",
        "hub_mass",
        "path_entropy",
    ]:
        for ref, other in comparisons:
            d = paired_deltas(
                panel_e, reference=ref, others=[other], metric=metric
            )
            delta_frames.append(d)
    deltas = pd.concat(delta_frames, ignore_index=True) if delta_frames else pd.DataFrame()
    deltas.to_csv(summary_dir / "paired_deltas.csv", index=False)

    wins = per_seed_wins(
        panel_e,
        reference="E2_hcr2",
        controls=["E4_patient_shuffle", "E5_edgewise_shuffle", "E1_hgt"],
        metric="reciprocal_rank_at_100",
    )
    wins.to_csv(summary_dir / "per_seed_wins.csv", index=False)

    decision = decide_wave5c(
        panel_t_sum,
        panel_e,
        selected_config={
            "temperature": selected.get("temperature"),
            "length_penalty": selected.get("length_penalty"),
            "uncertainty_penalty": selected.get("uncertainty_penalty", 0.0),
        },
    )
    # If E3 beats E2 on MRR, prefer its uncertainty penalty for 5C when Result A.
    e3_vs_e2 = deltas[
        (deltas["reference"] == "E3_hcr2_uncertainty")
        & (deltas["variant"] == "E2_hcr2")
        & (deltas["metric"] == "reciprocal_rank_at_100")
    ]
    if len(e3_vs_e2) and float(e3_vs_e2.iloc[0]["delta_mean"]) > 0:
        decision["selected_uncertainty_penalty"] = float(
            cfg["energy"]["uncertainty_penalties"][1]
            if len(cfg["energy"]["uncertainty_penalties"]) > 1
            else 0.25
        )
        if decision.get("result_a_pass"):
            decision["energy_mode_for_wave5c"] = "hcr_log_prob_with_uncertainty"

    write_json(summary_dir / "WAVE5B_DECISION.json", decision)

    print("=== PANEL T (pooled) ===")
    print(
        panel_t_sum.groupby("variant")[
            ["coverage", "true_path_coverage", "unconditional_tpm"]
        ]
        .mean()
        .to_string()
    )
    print("\n=== PANEL E test (pooled) ===")
    print(
        panel_e_sum.groupby("variant")[
            ["tpm", "mrr_at_100", "true_path_hit_at_5", "hub_mass"]
        ]
        .mean()
        .to_string()
    )
    print("\n=== PAIRED DELTAS (MRR@100) ===")
    print(
        deltas[deltas["metric"] == "reciprocal_rank_at_100"].to_string(index=False)
    )
    print("\n=== DECISION ===")
    print(json.dumps(decision, indent=2))
    print(f"\nWrote {summary_dir}")


if __name__ == "__main__":
    main()
