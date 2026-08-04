#!/usr/bin/env python3
"""Summarize Wave 5C ΔHEM panels and write WAVE5C_DECISION.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wnerw.wave5c.paired_metrics import (
    decide_wave5c,
    paired_delta_hem,
    per_seed_delta,
    variant_summary,
    write_json,
)
from wnerw.wave5c.runtime import load_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/wnerw/wave5c.yaml")
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    base = ROOT / cfg["paths"]["output_dir"]
    summary_dir = base / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    factual = pd.read_csv(base / "query_metrics" / "factual_query_patient_metrics.csv")
    paired_raw = pd.read_csv(
        base / "query_metrics" / "paired_toggle_query_patient_metrics.csv"
    )
    selected = json.loads(
        (base / "tuning" / "selected_patient_energy.json").read_text()
    )

    all_raw = pd.concat([factual, paired_raw], ignore_index=True)
    paired = paired_delta_hem(all_raw)
    paired.to_csv(base / "patient_pair_metrics" / "paired_delta_hem.csv", index=False) if False else None
    (base / "patient_pair_metrics").mkdir(parents=True, exist_ok=True)
    paired.to_csv(base / "patient_pair_metrics" / "paired_delta_hem.csv", index=False)

    vsum = variant_summary(paired)
    vsum.to_csv(summary_dir / "variant_summary.csv", index=False)

    seed_wins = per_seed_delta(paired)
    seed_wins.to_csv(summary_dir / "per_seed_wins.csv", index=False)

    # Paired deltas C2 vs controls on panel P
    p = paired[paired["panel"] == "P"]
    wide = p.pivot_table(
        index=["seed", "query_id"], columns="variant", values="delta_hem", aggfunc="mean"
    )
    delta_rows = []
    ref = "C2_structural_gate"
    for other in [
        "C1_node_activity",
        "C4_patient_shuffle",
        "C5_context_shuffle",
        "C6_matched_random_context",
        "C0_neutral",
        "C3_gate_plus_hcr_prior",
        "C7_hgt_topology",
    ]:
        if ref not in wide.columns or other not in wide.columns:
            continue
        d = (wide[ref] - wide[other]).dropna()
        delta_rows.append(
            {
                "reference": ref,
                "variant": other,
                "n_paired": int(len(d)),
                "delta_mean": float(d.mean()),
                "frac_ref_better": float((d > 0).mean()) if len(d) else float("nan"),
            }
        )
    deltas = pd.DataFrame(delta_rows)
    deltas.to_csv(summary_dir / "paired_deltas.csv", index=False)

    # Context specificity: C2 vs C5/C6
    spec = deltas[deltas["variant"].isin(["C5_context_shuffle", "C6_matched_random_context"])]
    spec.to_csv(summary_dir / "context_specificity.csv", index=False)

    hub_by_variant = (
        paired_raw.groupby("variant")["hub_mass"].mean().to_dict()
        if len(paired_raw)
        else {}
    )
    decision = decide_wave5c(
        paired, hub_by_variant, selected=selected, panel="P"
    )
    write_json(summary_dir / "WAVE5C_DECISION.json", decision)

    print("=== VARIANT SUMMARY ===")
    print(vsum.to_string(index=False))
    print("\n=== C2 vs controls (ΔHEM) ===")
    print(deltas.to_string(index=False))
    print("\n=== DECISION ===")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
