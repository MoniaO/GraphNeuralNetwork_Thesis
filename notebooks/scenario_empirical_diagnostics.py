#!/usr/bin/env python3
"""Scenario-specific empirical diagnostics for the v2.1 pharmacotherapy benchmark.

Two layers:
  1) Static audited graph is shared across scenarios (not recomputed here).
  2) Patient data and HCR/CMI support differ by scenario — this script.

Outputs under experiments/scenario_empirical_diagnostics_outputs/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SCENARIO_ORDER = [
    "clean",
    "hidden_confounder",
    "selection_bias",
    "no_overlap",
    "noisy_documentation",
    "multihospital",
]

DROP_META = {
    "patient_id",
    "combined_record_id",
    "source_file",
    "generation_seed",
    "paired_patient_key",
    "dataset_scenario",
    "benchmark_scenario",
    "benchmark_split",
    "replicate_id",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("synthetic_pharmacotherapy_v2_1_nn_benchmark"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/scenario_empirical_diagnostics_outputs"),
    )
    return p.parse_args()


def prevalence(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce")
    if x.notna().sum() == 0:
        return float("nan")
    # binary-ish: treat >0 as present when values in {0,1}
    vals = x.dropna()
    if vals.nunique() <= 2 and vals.min() >= 0 and vals.max() <= 1:
        return float(vals.mean())
    return float((vals != 0).mean())


def load_graph_meta(benchmark_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_csv(benchmark_dir / "synthetic_pharmacotherapy_v2_nodes.csv")
    edges = pd.read_csv(benchmark_dir / "synthetic_pharmacotherapy_v2_1_edges_audited.csv")
    return nodes, edges


def scenario_patient_summary(
    patients: pd.DataFrame,
    nodes: pd.DataFrame,
) -> pd.DataFrame:
    by_type = nodes.groupby("node_type")["node"].apply(list).to_dict()
    rows = []
    for scenario, g in patients.groupby("benchmark_scenario", sort=False):
        row: dict[str, object] = {
            "benchmark_scenario": scenario,
            "n_patients": len(g),
            "n_replicates": int(g["replicate_id"].nunique()) if "replicate_id" in g else np.nan,
        }
        if "benchmark_split" in g.columns:
            for split, n in g["benchmark_split"].value_counts().items():
                row[f"n_{split}"] = int(n)
        # missingness: fraction of columns with any NaN + mean NaN rate over node cols
        node_cols = [c for c in nodes["node"] if c in g.columns]
        if node_cols:
            miss = g[node_cols].isna()
            row["missing_rate_mean"] = float(miss.mean().mean())
            row["cols_with_any_missing"] = int((miss.any()).sum())
            row["unobserved_severity_available"] = (
                "unobserved_severity" in g.columns
                and float(g["unobserved_severity"].notna().mean()) > 0
            )
            row["unobserved_severity_nonnull_rate"] = (
                float(g["unobserved_severity"].notna().mean())
                if "unobserved_severity" in g.columns
                else 0.0
            )
        # endpoint prevalences
        for node in by_type.get("clinical_endpoint", []):
            if node in g.columns:
                row[f"prev_{node}"] = prevalence(g[node])
        # drug mean exposure
        drug_cols = [c for c in by_type.get("drug_exposure", []) if c in g.columns]
        if drug_cols:
            row["mean_drug_prevalence"] = float(np.nanmean([prevalence(g[c]) for c in drug_cols]))
        mech_cols = [c for c in by_type.get("mechanism", []) if c in g.columns]
        if mech_cols:
            row["mean_mechanism_prevalence"] = float(
                np.nanmean([prevalence(g[c]) for c in mech_cols])
            )
        obs_cols = [c for c in by_type.get("observation_or_selection", []) if c in g.columns]
        if obs_cols:
            row["mean_observation_prevalence"] = float(
                np.nanmean([prevalence(g[c]) for c in obs_cols])
            )
        # scenario-specific markers
        for col in [
            "hospital_contact",
            "monitoring_intensity",
            "creatinine_tested",
            "lft_tested",
            "adr_reported",
            "true_AKI",
            "recorded_AKI",
            "AKI",
        ]:
            if col in g.columns and g[col].notna().any():
                row[f"prev_{col}"] = prevalence(g[col])
        if "hospital_id" in g.columns:
            row["n_hospitals"] = int(g["hospital_id"].nunique())
            row["hospitals"] = ",".join(str(int(x)) for x in sorted(g["hospital_id"].dropna().unique()))
        rows.append(row)
    out = pd.DataFrame(rows)
    out["benchmark_scenario"] = pd.Categorical(out["benchmark_scenario"], SCENARIO_ORDER, ordered=True)
    return out.sort_values("benchmark_scenario")


def endpoint_prevalence_long(patients: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    endpoints = nodes.loc[nodes["node_type"] == "clinical_endpoint", "node"].tolist()
    rows = []
    for scenario, g in patients.groupby("benchmark_scenario"):
        for ep in endpoints:
            if ep not in g.columns:
                continue
            rows.append(
                {
                    "benchmark_scenario": scenario,
                    "endpoint": ep,
                    "prevalence": prevalence(g[ep]),
                }
            )
        # noisy documentation extras
        for ep in endpoints:
            tcol, rcol = f"true_{ep}", f"recorded_{ep}"
            if tcol in g.columns and g[tcol].notna().any():
                rows.append(
                    {
                        "benchmark_scenario": scenario,
                        "endpoint": tcol,
                        "prevalence": prevalence(g[tcol]),
                    }
                )
            if rcol in g.columns and g[rcol].notna().any():
                rows.append(
                    {
                        "benchmark_scenario": scenario,
                        "endpoint": rcol,
                        "prevalence": prevalence(g[rcol]),
                    }
                )
    return pd.DataFrame(rows)


def observation_marker_long(patients: pd.DataFrame) -> pd.DataFrame:
    markers = [
        "hospital_contact",
        "monitoring_intensity",
        "creatinine_tested",
        "lft_tested",
        "adr_reported",
        "electrolytes_tested",
        "ecg_performed",
    ]
    rows = []
    for scenario, g in patients.groupby("benchmark_scenario"):
        for m in markers:
            if m in g.columns and g[m].notna().any():
                rows.append(
                    {
                        "benchmark_scenario": scenario,
                        "marker": m,
                        "prevalence": prevalence(g[m]),
                    }
                )
    return pd.DataFrame(rows)


def hcr_support_tables(
    features: pd.DataFrame,
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    obs_ids = set(candidates["candidate_id"])
    feat = features[features["candidate_id"].isin(obs_ids)].copy()
    meta_cols = ["candidate_id", "target_edge_type", "layer_transition", "source", "target"]
    meta = candidates[meta_cols].drop_duplicates("candidate_id")
    # features already has source/target — keep candidate meta without suffix collisions
    add_cols = [c for c in meta_cols if c == "candidate_id" or c not in feat.columns]
    feat = feat.merge(meta[add_cols], on="candidate_id", how="left")
    if "layer_transition" not in feat.columns:
        feat = feat.merge(
            candidates[["candidate_id", "layer_transition"]].drop_duplicates("candidate_id"),
            on="candidate_id",
            how="left",
        )
    if "target_edge_type" not in feat.columns:
        feat = feat.merge(
            candidates[["candidate_id", "target_edge_type"]].drop_duplicates("candidate_id"),
            on="candidate_id",
            how="left",
        )

    by_label = (
        feat.groupby(["dataset_scenario", "edge_label"], as_index=False)
        .agg(
            n=("candidate_id", "count"),
            hcr_median=("hcr_information_weight", "median"),
            hcr_mean=("hcr_information_weight", "mean"),
            cmi_median=("conditional_mutual_information", "median"),
            p_perm_median=("permutation_p_value", "median"),
            above_null_rate=("hcr_above_null_q95", "mean"),
        )
    )

    # positives only by edge type
    pos = feat[feat["edge_label"] == 1]
    by_etype = (
        pos.groupby(["dataset_scenario", "target_edge_type"], as_index=False)
        .agg(
            n=("candidate_id", "count"),
            hcr_median=("hcr_information_weight", "median"),
            cmi_median=("conditional_mutual_information", "median"),
            above_null_rate=("hcr_above_null_q95", "mean"),
        )
    )

    # non-edges: observation/selection vs other
    neg = feat[feat["edge_label"] == 0].copy()
    src = neg["source"].astype(str) if "source" in neg.columns else pd.Series("", index=neg.index)
    tgt = neg["target"].astype(str) if "target" in neg.columns else pd.Series("", index=neg.index)
    lt = (
        neg["layer_transition"].astype(str)
        if "layer_transition" in neg.columns
        else pd.Series("", index=neg.index)
    )
    obs_pat = "hospital_contact|tested|reported|recorded|monitoring"
    neg["family"] = np.where(
        lt.str.contains("observation|selection", case=False, na=False)
        | src.str.contains(obs_pat, case=False, na=False)
        | tgt.str.contains(obs_pat, case=False, na=False),
        "observation_selection",
        "other",
    )
    by_neg_family = (
        neg.groupby(["dataset_scenario", "family"], as_index=False)
        .agg(
            n=("candidate_id", "count"),
            hcr_median=("hcr_information_weight", "median"),
            above_null_rate=("hcr_above_null_q95", "mean"),
        )
    )
    return by_label, by_etype, by_neg_family


def save_figs(
    out: Path,
    patient_sum: pd.DataFrame,
    ep_long: pd.DataFrame,
    obs_long: pd.DataFrame,
    by_label: pd.DataFrame,
    by_etype: pd.DataFrame,
    by_neg_family: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid")
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1) n patients / splits
    split_cols = [c for c in patient_sum.columns if c.startswith("n_") and c != "n_patients" and c != "n_replicates" and c != "n_hospitals"]
    if split_cols:
        long = patient_sum.melt(
            id_vars=["benchmark_scenario"],
            value_vars=split_cols,
            var_name="split",
            value_name="n",
        )
        long["split"] = long["split"].str.replace("^n_", "", regex=True)
        fig, ax = plt.subplots(figsize=(11, 4.5))
        sns.barplot(data=long, x="benchmark_scenario", y="n", hue="split", ax=ax)
        ax.set_title("Patients per scenario × split")
        ax.tick_params(axis="x", rotation=25)
        plt.tight_layout()
        fig.savefig(fig_dir / "01_patients_by_split.png", dpi=150)
        plt.close(fig)

    # 2) endpoint prevalence heatmap (clinical endpoints only)
    ep_clin = ep_long[~ep_long["endpoint"].str.startswith(("true_", "recorded_"))].copy()
    if len(ep_clin):
        heat = ep_clin.pivot(index="endpoint", columns="benchmark_scenario", values="prevalence")
        heat = heat.reindex(columns=[c for c in SCENARIO_ORDER if c in heat.columns])
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(heat, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax)
        ax.set_title("Endpoint prevalence by scenario")
        plt.tight_layout()
        fig.savefig(fig_dir / "02_endpoint_prevalence_heatmap.png", dpi=150)
        plt.close(fig)

    # 3) observation markers
    if len(obs_long):
        fig, ax = plt.subplots(figsize=(11, 4.5))
        sns.barplot(data=obs_long, x="benchmark_scenario", y="prevalence", hue="marker", ax=ax)
        ax.set_title("Observation / selection markers by scenario")
        ax.tick_params(axis="x", rotation=25)
        ax.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        fig.savefig(fig_dir / "03_observation_markers.png", dpi=150)
        plt.close(fig)

    # 4) HCR by label
    plot = by_label.copy()
    plot["edge_label"] = plot["edge_label"].astype(int)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.barplot(
        data=plot,
        x="dataset_scenario",
        y="hcr_median",
        hue="edge_label",
        order=[s for s in SCENARIO_ORDER if s in set(plot["dataset_scenario"])],
        ax=ax,
    )
    ax.set_title("Median HCR: edge_label 0 vs 1 (observed candidates)")
    ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    fig.savefig(fig_dir / "04_hcr_by_label.png", dpi=150)
    plt.close(fig)

    # 5) HCR by edge type (positives), focus scenarios
    focus_types = [
        "drug_to_mechanism",
        "mechanism_to_adr",
        "adr_to_endpoint",
        "observation",
        "observation_process",
        "selection",
        "confounding",
        "latent_confounding",
        "causal_mechanistic",
    ]
    et = by_etype[by_etype["target_edge_type"].isin(focus_types)].copy()
    if len(et):
        heat = et.pivot(index="target_edge_type", columns="dataset_scenario", values="hcr_median")
        heat = heat.reindex(columns=[c for c in SCENARIO_ORDER if c in heat.columns])
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(heat, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax)
        ax.set_title("Median HCR for TRUE edges by edge_type × scenario")
        plt.tight_layout()
        fig.savefig(fig_dir / "05_hcr_true_by_edge_type.png", dpi=150)
        plt.close(fig)

    # 6) non-edge HCR families
    if len(by_neg_family):
        fig, ax = plt.subplots(figsize=(10, 4.5))
        sns.barplot(
            data=by_neg_family,
            x="dataset_scenario",
            y="hcr_median",
            hue="family",
            order=[s for s in SCENARIO_ORDER if s in set(by_neg_family["dataset_scenario"])],
            ax=ax,
        )
        ax.set_title("Median HCR for NON-edges: observation/selection vs other")
        ax.tick_params(axis="x", rotation=25)
        plt.tight_layout()
        fig.savefig(fig_dir / "06_hcr_nonedge_families.png", dpi=150)
        plt.close(fig)

    # 7) noisy true vs recorded AKI if present
    noisy = ep_long[
        (ep_long["benchmark_scenario"] == "noisy_documentation")
        & (ep_long["endpoint"].isin(["true_AKI", "recorded_AKI", "AKI"]))
    ]
    if len(noisy):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.barplot(data=noisy, x="endpoint", y="prevalence", ax=ax, color="#457b9d")
        ax.set_title("noisy_documentation: AKI true / recorded / observed")
        plt.tight_layout()
        fig.savefig(fig_dir / "07_noisy_aki_true_vs_recorded.png", dpi=150)
        plt.close(fig)


def write_notes(out: Path, patient_sum: pd.DataFrame) -> None:
    lines = [
        "# Scenario empirical diagnostics — notes",
        "",
        "## Static vs scenario-specific",
        "",
        "- Audited graph (nodes/edges/types/layers) is **shared** across scenarios.",
        "- Patient distributions, endpoint prevalences, observability and HCR/CMI **change** by scenario.",
        "",
        "## Scenario markers observed in this run",
        "",
    ]
    for _, r in patient_sum.iterrows():
        s = r["benchmark_scenario"]
        lines.append(f"### {s}")
        lines.append(f"- n_patients = {r.get('n_patients')}")
        lines.append(
            f"- unobserved_severity available = {r.get('unobserved_severity_available')} "
            f"(nonnull rate={r.get('unobserved_severity_nonnull_rate')})"
        )
        if "prev_hospital_contact" in r and pd.notna(r.get("prev_hospital_contact")):
            lines.append(f"- hospital_contact prevalence = {r['prev_hospital_contact']:.4f}")
        if "prev_true_AKI" in r and pd.notna(r.get("prev_true_AKI")):
            lines.append(f"- true_AKI = {r['prev_true_AKI']:.4f}")
        if "prev_recorded_AKI" in r and pd.notna(r.get("prev_recorded_AKI")):
            lines.append(f"- recorded_AKI = {r['prev_recorded_AKI']:.4f}")
        if "hospitals" in r and pd.notna(r.get("hospitals")):
            lines.append(f"- hospitals = {r['hospitals']}")
        lines.append("")
    lines += [
        "## Modelling reminders",
        "",
        "- Link prediction target = `edge_label`, not AKI.",
        "- In `hidden_confounder` do **not** use `unobserved_severity` as a feature (column is all-NaN here).",
        "- Use `edge_splits_observed` + train-only HCR features.",
        "- Structure learning (NOTEARS/PC/FCI) is the next layer after these diagnostics.",
        "",
    ]
    (out / "diagnostic_notes.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    b = args.benchmark_dir
    hcr = b / "hcr_link_prediction"

    nodes, edges = load_graph_meta(b)
    patients = pd.read_csv(b / "processed_v2_1" / "synthetic_pharmacotherapy_v2_1_samples_combined.csv")
    candidates = pd.read_csv(hcr / "edge_candidates_observed.csv")
    features = pd.read_csv(hcr / "edge_pair_features_train_only.csv")

    static = {
        "n_nodes": int(len(nodes)),
        "n_edges": int(len(edges)),
        "n_node_types": int(nodes["node_type"].nunique()),
        "n_edge_types": int(edges["edge_type"].nunique()) if "edge_type" in edges.columns else None,
        "observed_candidates": int(len(candidates)),
        "observed_positives": int((candidates["edge_label"] == 1).sum()),
        "observed_negatives": int((candidates["edge_label"] == 0).sum()),
        "note": "Audited graph is shared; only patient/HCR layers differ by scenario.",
    }
    (out / "static_graph_pointer.json").write_text(json.dumps(static, indent=2), encoding="utf-8")

    patient_sum = scenario_patient_summary(patients, nodes)
    ep_long = endpoint_prevalence_long(patients, nodes)
    obs_long = observation_marker_long(patients)
    by_label, by_etype, by_neg_family = hcr_support_tables(features, candidates)

    patient_sum.to_csv(out / "scenario_patient_summary.csv", index=False)
    ep_long.to_csv(out / "endpoint_prevalence_long.csv", index=False)
    obs_long.to_csv(out / "observation_marker_prevalence.csv", index=False)
    by_label.to_csv(out / "hcr_support_by_label.csv", index=False)
    by_etype.to_csv(out / "hcr_support_by_edge_type.csv", index=False)
    by_neg_family.to_csv(out / "hcr_support_nonedge_families.csv", index=False)

    # drug prevalences (compact)
    drugs = nodes.loc[nodes["node_type"] == "drug_exposure", "node"].tolist()
    drug_rows = []
    for scenario, g in patients.groupby("benchmark_scenario"):
        for d in drugs:
            if d in g.columns:
                drug_rows.append(
                    {"benchmark_scenario": scenario, "drug": d, "prevalence": prevalence(g[d])}
                )
    pd.DataFrame(drug_rows).to_csv(out / "drug_prevalence_long.csv", index=False)

    save_figs(out, patient_sum, ep_long, obs_long, by_label, by_etype, by_neg_family)
    write_notes(out, patient_sum)

    print("Wrote:", out)
    print(patient_sum[["benchmark_scenario", "n_patients", "unobserved_severity_nonnull_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
