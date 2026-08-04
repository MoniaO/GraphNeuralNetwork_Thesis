#!/usr/bin/env python3
"""Tune (α, η) on validation paired toggles for C2; freeze T/λ_L from Wave 5B."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wnerw.wave5c.paired_metrics import paired_delta_hem
from wnerw.wave5c.patient_evaluator import evaluate_patient_query
from wnerw.wave5c.patient_energy import PatientEnergyConfig
from wnerw.wave5c.patient_queries import parse_context_nodes
from wnerw.wave5c.runtime import load_cfg, load_seed_bundle, patient_activity_for


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/wnerw/wave5c.yaml")
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    out = ROOT / cfg["paths"]["output_dir"] / "tuning"
    out.mkdir(parents=True, exist_ok=True)

    queries = pd.read_csv(ROOT / cfg["paths"]["output_dir"] / "registry" / "gate_queries.csv")
    pairs = pd.read_csv(
        ROOT / cfg["paths"]["output_dir"] / "registry" / "patient_pair_assignments.csv"
    )
    # Tune on Panel P valid queries only.
    q_valid = queries[queries["query_split"] == cfg["evaluation"]["tune_split"]]
    pairs = pairs[(pairs["panel"] == "P") & (pairs["query_id"].isin(set(q_valid["query_id"])))]

    alphas = list(cfg["energy"]["node_activity_grid"])
    etas = list(cfg["energy"]["gate_weight_grid"])
    grid_rows = []

    for alpha, eta in product(alphas, etas):
        rows = []
        hub_c2, hub_c1 = [], []
        for seed in cfg["seeds"]:
            bundle = load_seed_bundle(cfg, int(seed))
            qseed = q_valid[q_valid["seed"] == int(seed)]
            pseed = pairs[pairs["seed"] == int(seed)]
            if qseed.empty or pseed.empty:
                continue
            qmap = qseed.set_index("query_id")
            cfg_c2 = PatientEnergyConfig(
                temperature=0.5,
                length_penalty=0.1,
                node_activity_weight=float(alpha),
                gate_weight=float(eta),
                static_hcr_weight=0.0,
            )
            cfg_c1 = PatientEnergyConfig(
                temperature=0.5,
                length_penalty=0.1,
                node_activity_weight=float(alpha),
                gate_weight=0.0,
                static_hcr_weight=0.0,
            )
            for _, asg in pseed.iterrows():
                qid = str(asg["query_id"])
                if qid not in qmap.index:
                    continue
                q = qmap.loc[qid]
                if isinstance(q, pd.DataFrame):
                    q = q.iloc[0]
                query = q.to_dict()
                query["query_id"] = qid
                if str(asg.get("profile_mode")) == "toggle" and pd.notna(
                    asg.get("profile_json")
                ):
                    activity = {k: float(v) for k, v in json.loads(asg["profile_json"]).items()}
                    pid = str(asg["patient_id"])
                else:
                    pid = str(asg["patient_id"]).split("#")[0]
                    activity = patient_activity_for(
                        bundle["patients"], pid, bundle["admissible"]
                    )
                m2 = evaluate_patient_query(
                    pid,
                    activity,
                    query,
                    bundle["graph_hcr"],
                    bundle["edges"],
                    bundle["context_map"],
                    bundle["admissible"],
                    bundle["hubs"],
                    cfg_c2,
                    skip_topk=True,
                )
                m1 = evaluate_patient_query(
                    pid,
                    activity,
                    query,
                    bundle["graph_hcr"],
                    bundle["edges"],
                    bundle["context_map"],
                    bundle["admissible"],
                    bundle["hubs"],
                    cfg_c1,
                    skip_topk=True,
                )
                rows.append(
                    {
                        "variant": "C2_structural_gate",
                        "seed": int(seed),
                        "query_id": qid,
                        "panel": "P",
                        "gate_state": str(asg["gate_state"]),
                        "hidden_edge_mass": float(m2["hidden_edge_mass"]),
                    }
                )
                hub_c2.append(float(m2["hub_mass"]))
                hub_c1.append(float(m1["hub_mass"]))

        paired = paired_delta_hem(pd.DataFrame(rows)) if rows else pd.DataFrame()
        mean_delta = float(paired["delta_hem"].mean()) if len(paired) else float("-inf")
        n_pos_seed = (
            int(
                paired.groupby("seed")["delta_hem"]
                .mean()
                .gt(0)
                .sum()
            )
            if len(paired)
            else 0
        )
        hub_diff = (
            (sum(hub_c2) / len(hub_c2) - sum(hub_c1) / len(hub_c1))
            if hub_c2 and hub_c1
            else 0.0
        )
        rejected = hub_diff > 0.05
        grid_rows.append(
            {
                "node_activity_weight": float(alpha),
                "gate_weight": float(eta),
                "delta_hem_mean": mean_delta,
                "n_seeds_positive": n_pos_seed,
                "hub_mass_diff_c2_c1": float(hub_diff),
                "rejected_hub_mass": rejected,
                "n_paired": int(len(paired)),
            }
        )
        print(
            f"α={alpha} η={eta} ΔHEM={mean_delta:.4f} "
            f"seeds+={n_pos_seed} rejected={rejected}",
            flush=True,
        )

    grid = pd.DataFrame(grid_rows)
    grid.to_csv(out / "validation_grid.csv", index=False)
    cand = grid[(~grid["rejected_hub_mass"]) & (grid["n_seeds_positive"] >= 4)]
    if cand.empty:
        cand = grid[~grid["rejected_hub_mass"]]
    if cand.empty:
        cand = grid
    cand = cand.sort_values(
        by=["delta_hem_mean", "gate_weight"],
        ascending=[False, True],
    )
    best = cand.iloc[0]
    selected = {
        "node_activity_weight": float(best["node_activity_weight"]),
        "gate_weight": float(best["gate_weight"]),
        "static_hcr_prior_weight": 0.0,
        "temperature": 0.5,
        "length_penalty": 0.1,
        "uncertainty_penalty": 0.0,
        "validation_delta_hem": float(best["delta_hem_mean"]),
        "n_seeds_positive": int(best["n_seeds_positive"]),
    }
    (out / "selected_patient_energy.json").write_text(json.dumps(selected, indent=2))
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
