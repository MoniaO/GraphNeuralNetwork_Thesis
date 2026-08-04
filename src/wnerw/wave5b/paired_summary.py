"""Paired Panel summaries and Wave-5C gate decision."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def summarize_panel_t(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, seed), sub in df.groupby(["variant", "seed"]):
        rows.append(
            {
                "variant": variant,
                "seed": int(seed),
                "coverage": float(sub["has_any_path"].mean()),
                "true_path_coverage": float(sub["has_true_path"].mean()),
                "unconditional_tpm": float(sub["true_path_mass"].mean()),
                "hidden_edge_hit_at_5": float(sub["hidden_edge_hit_at_5"].mean()),
                "n_paths_mean": float(sub["n_paths"].mean()),
                "n_queries": int(len(sub)),
            }
        )
    return pd.DataFrame(rows).sort_values(["variant", "seed"])


def summarize_panel_e(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, seed), sub in df.groupby(["variant", "seed"]):
        rows.append(
            {
                "variant": variant,
                "seed": int(seed),
                "coverage": float(sub["has_any_path"].mean()),
                "tpm": float(sub["true_path_mass"].mean()),
                "mrr_at_100": float(sub["reciprocal_rank_at_100"].mean()),
                "true_path_hit_at_1": float(sub["true_path_hit_at_1"].mean()),
                "true_path_hit_at_5": float(sub["true_path_hit_at_5"].mean()),
                "hidden_edge_hit_at_5": float(sub["hidden_edge_hit_at_5"].mean()),
                "path_entropy": float(sub["path_entropy"].mean()),
                "hub_mass": float(sub["hub_mass"].mean()),
                "n_queries": int(len(sub)),
            }
        )
    return pd.DataFrame(rows).sort_values(["variant", "seed"])


def paired_deltas(
    df: pd.DataFrame,
    *,
    reference: str,
    others: list[str],
    metric: str = "reciprocal_rank_at_100",
) -> pd.DataFrame:
    wide = df.pivot_table(
        index=["seed", "query_id"],
        columns="variant",
        values=metric,
        aggfunc="first",
    )
    rows = []
    for other in others:
        if reference not in wide.columns or other not in wide.columns:
            continue
        d = (wide[reference] - wide[other]).dropna()
        rows.append(
            {
                "reference": reference,
                "variant": other,
                "metric": metric,
                "n_paired": int(len(d)),
                "delta_mean": float(d.mean()),
                "delta_std": float(d.std(ddof=0)),
                "frac_ref_better": float((d > 0).mean()) if len(d) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def per_seed_wins(
    df: pd.DataFrame,
    *,
    reference: str,
    controls: list[str],
    metric: str = "reciprocal_rank_at_100",
) -> pd.DataFrame:
    rows = []
    for seed, sub in df.groupby("seed"):
        wide = sub.pivot_table(
            index="query_id", columns="variant", values=metric, aggfunc="first"
        )
        if reference not in wide.columns:
            continue
        for ctrl in controls:
            if ctrl not in wide.columns:
                continue
            d = (wide[reference] - wide[ctrl]).dropna()
            rows.append(
                {
                    "seed": int(seed),
                    "reference": reference,
                    "control": ctrl,
                    "metric": metric,
                    "delta_mean": float(d.mean()),
                    "ref_wins": bool(d.mean() > 0),
                }
            )
    return pd.DataFrame(rows)


def decide_wave5c(
    panel_t_summary: pd.DataFrame,
    panel_e_query: pd.DataFrame,
    *,
    selected_config: dict,
    hub_mass_slack: float = 0.05,
) -> dict:
    """Apply Result A / Result B gate from the Wave-5B protocol."""
    ref = "E2_hcr2"
    controls = ["E4_patient_shuffle", "E5_edgewise_shuffle"]
    wins = per_seed_wins(
        panel_e_query, reference=ref, controls=controls, metric="reciprocal_rank_at_100"
    )
    hit_wins = per_seed_wins(
        panel_e_query, reference=ref, controls=controls, metric="true_path_hit_at_5"
    )

    n_seed_mrr_ok = 0
    seeds = sorted(panel_e_query["seed"].unique())
    for seed in seeds:
        seed_ok = True
        for ctrl in controls:
            sub = wins[(wins["seed"] == seed) & (wins["control"] == ctrl)]
            if sub.empty or not bool(sub.iloc[0]["ref_wins"]):
                seed_ok = False
                break
        if seed_ok:
            n_seed_mrr_ok += 1

    deltas_mrr = paired_deltas(
        panel_e_query, reference=ref, others=controls, metric="reciprocal_rank_at_100"
    )
    deltas_hit = paired_deltas(
        panel_e_query, reference=ref, others=controls, metric="true_path_hit_at_5"
    )
    deltas_hub = paired_deltas(
        panel_e_query, reference=ref, others=["E1_hgt"], metric="hub_mass"
    )

    mean_mrr_deltas_positive = (
        bool((deltas_mrr["delta_mean"] > 0).all()) if len(deltas_mrr) else False
    )
    mean_hit_deltas_positive = (
        bool((deltas_hit["delta_mean"] > 0).all()) if len(deltas_hit) else False
    )
    hub_ok = True
    if len(deltas_hub):
        # HCR hub mass should not exceed HGT by more than slack (delta = HCR - HGT).
        hub_ok = float(deltas_hub.iloc[0]["delta_mean"]) <= hub_mass_slack

    # Topology claim from Panel T pooled.
    t_pool = (
        panel_t_summary.groupby("variant")[["coverage", "true_path_coverage"]]
        .mean()
        .reset_index()
    )
    t_hcr = t_pool.loc[t_pool["variant"] == "T2_hcr"].iloc[0] if (
        t_pool["variant"] == "T2_hcr"
    ).any() else None
    t_hgt = t_pool.loc[t_pool["variant"] == "T1_hgt"].iloc[0] if (
        t_pool["variant"] == "T1_hgt"
    ).any() else None
    topology_improves = bool(
        t_hcr is not None
        and t_hgt is not None
        and float(t_hcr["true_path_coverage"]) > float(t_hgt["true_path_coverage"])
    )

    result_a = (
        n_seed_mrr_ok >= 4
        and mean_mrr_deltas_positive
        and mean_hit_deltas_positive
        and hub_ok
    )

    if result_a:
        energy_result = "HCR energy beats patient-shuffle and edgewise-shuffle"
        energy_mode = "hcr_log_prob"
    else:
        energy_result = (
            "HCR improves graph reconstruction / reachability, "
            "but unique energy advantage not shown"
        )
        energy_mode = "topology_only_neutral_or_tuned_weights"

    decision = {
        "wave": "WAVE5B",
        "topology_result": (
            "HCR improves true-path coverage"
            if topology_improves
            else "HCR topology advantage not shown"
        ),
        "energy_result": energy_result,
        "selected_temperature": selected_config.get("temperature"),
        "selected_length_penalty": selected_config.get("length_penalty"),
        "selected_uncertainty_penalty": selected_config.get("uncertainty_penalty"),
        "energy_mode_for_wave5c": energy_mode,
        "n_seeds_mrr_beats_both_controls": int(n_seed_mrr_ok),
        "mean_mrr_deltas_positive": mean_mrr_deltas_positive,
        "mean_hit5_deltas_positive": mean_hit_deltas_positive,
        "hub_mass_ok_vs_hgt": hub_ok,
        "result_a_pass": result_a,
        "result_label": "A" if result_a else "B",
    }
    return decision


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
