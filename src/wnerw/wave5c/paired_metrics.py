"""Paired ΔHEM summaries and Wave 5C decision."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def paired_delta_hem(df: pd.DataFrame) -> pd.DataFrame:
    """Expect columns: variant, seed, query_id, panel, gate_state, hidden_edge_mass."""
    rows = []
    for keys, sub in df.groupby(["variant", "seed", "query_id", "panel"]):
        on = sub.loc[sub["gate_state"] == "ON", "hidden_edge_mass"]
        off = sub.loc[sub["gate_state"] == "OFF", "hidden_edge_mass"]
        if on.empty or off.empty:
            continue
        delta = float(on.mean() - off.mean())
        rows.append(
            {
                "variant": keys[0],
                "seed": int(keys[1]),
                "query_id": keys[2],
                "panel": keys[3],
                "hem_on": float(on.mean()),
                "hem_off": float(off.mean()),
                "delta_hem": delta,
                "consistency_hit": int(delta > 0),
            }
        )
    return pd.DataFrame(rows)


def variant_summary(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, panel), sub in paired.groupby(["variant", "panel"]):
        rows.append(
            {
                "variant": variant,
                "panel": panel,
                "n_query_seed": int(len(sub)),
                "delta_hem_mean": float(sub["delta_hem"].mean()),
                "delta_hem_std": float(sub["delta_hem"].std(ddof=0)),
                "consistency": float(sub["consistency_hit"].mean()),
                "hem_on_mean": float(sub["hem_on"].mean()),
                "hem_off_mean": float(sub["hem_off"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["panel", "delta_hem_mean"], ascending=[True, False])


def per_seed_delta(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, seed, panel), sub in paired.groupby(["variant", "seed", "panel"]):
        rows.append(
            {
                "variant": variant,
                "seed": int(seed),
                "panel": panel,
                "delta_hem_mean": float(sub["delta_hem"].mean()),
                "consistency": float(sub["consistency_hit"].mean()),
                "positive": bool(sub["delta_hem"].mean() > 0),
            }
        )
    return pd.DataFrame(rows)


def decide_wave5c(
    paired: pd.DataFrame,
    hub_by_variant: dict[str, float],
    *,
    selected: dict,
    panel: str = "P",
) -> dict:
    sub = paired[paired["panel"] == panel]
    seed_stats = per_seed_delta(sub)
    c2 = seed_stats[seed_stats["variant"] == "C2_structural_gate"]
    n_pos = int(c2["positive"].sum()) if len(c2) else 0

    def mean_delta(variant: str) -> float:
        s = sub[sub["variant"] == variant]
        return float(s["delta_hem"].mean()) if len(s) else float("nan")

    d_c2 = mean_delta("C2_structural_gate")
    checks = {
        "C2_gt_C1": d_c2 > mean_delta("C1_node_activity"),
        "C2_gt_C4": d_c2 > mean_delta("C4_patient_shuffle"),
        "C2_gt_C5": d_c2 > mean_delta("C5_context_shuffle"),
        "C2_gt_C6": d_c2 > mean_delta("C6_matched_random_context"),
    }
    consistency = float(
        sub.loc[sub["variant"] == "C2_structural_gate", "consistency_hit"].mean()
    ) if (sub["variant"] == "C2_structural_gate").any() else 0.0

    hub_c2 = float(hub_by_variant.get("C2_structural_gate", 0.0))
    hub_c1 = float(hub_by_variant.get("C1_node_activity", 0.0))
    hub_ok = (hub_c2 - hub_c1) <= 0.05

    pass_all = (
        n_pos >= 4
        and d_c2 > 0
        and all(checks.values())
        and consistency > 0.70
        and hub_ok
    )

    # C3 ablation: keep beta=0 unless C3 clearly better.
    d_c3 = mean_delta("C3_gate_plus_hcr_prior")
    beta = 0.0
    if d_c3 > d_c2 + 1e-6:
        beta = float(selected.get("static_hcr_prior_weight", 0.1))

    return {
        "wave": "WAVE5C",
        "topology": "HCR-selected G*",
        "temperature": 0.5,
        "length_penalty": 0.1,
        "uncertainty_penalty": 0.0,
        "selected_node_activity_weight": selected.get("node_activity_weight"),
        "selected_gate_weight": selected.get("gate_weight"),
        "static_hcr_prior_weight": beta,
        "primary_metric": "paired_delta_hidden_edge_mass",
        "delta_hem_c2": d_c2,
        "consistency_c2": consistency,
        "n_seeds_positive_c2": n_pos,
        "hub_mass_ok": hub_ok,
        "checks": checks,
        "patient_conditioning_confirmed": bool(pass_all),
        "result_label": "PASS" if pass_all else "FAIL",
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
