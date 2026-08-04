#!/usr/bin/env python3
"""Run Wave 5C patient-conditioned WNERW variants (factual + paired panels)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wnerw.wave5c.patient_activity import (
    make_gate_toggle_profiles,
    make_triple_off_profiles,
)
from wnerw.wave5c.patient_evaluator import evaluate_patient_query
from wnerw.wave5c.patient_queries import parse_context_nodes
from wnerw.wave5c.runtime import (
    energy_variants,
    load_cfg,
    load_seed_bundle,
    matched_random_context_map,
    patient_activity_for,
    shuffle_patient_profiles,
)


def resolve_activity(bundle, asg, query, *, shuffle_map=None):
    src = str(query["source"])
    ctx = parse_context_nodes(query.get("context_nodes"))
    needed = bundle["admissible"] | {src} | set(ctx)

    if str(asg.get("profile_mode")) == "toggle" and pd.notna(asg.get("profile_json")):
        # C4: rebuild toggle from a shuffled base patient (destroys pairing).
        if shuffle_map is not None:
            base_pid = str(asg.get("base_patient_id", asg["patient_id"])).split("#")[0]
            mapped = shuffle_map.get(base_pid, base_pid)
            base_act = patient_activity_for(bundle["patients"], mapped, needed)
            if len(ctx) >= 2 and str(asg["gate_state"]) == "OFF":
                # Which OFF variant is encoded in patient_id suffix #offj
                pid = str(asg["patient_id"])
                idx = 0
                if "#off" in pid:
                    try:
                        idx = int(pid.split("#off")[-1])
                    except ValueError:
                        idx = 0
                offs = make_triple_off_profiles(base_act, src, ctx)
                return offs[min(idx, len(offs) - 1)]
            on_prof, off_prof = make_gate_toggle_profiles(base_act, src, ctx)
            return on_prof if str(asg["gate_state"]) == "ON" else off_prof
        return {k: float(v) for k, v in json.loads(asg["profile_json"]).items()}

    pid = str(asg["patient_id"]).split("#")[0]
    if shuffle_map is not None:
        pid = shuffle_map.get(pid, pid)
    return patient_activity_for(bundle["patients"], pid, needed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/wnerw/wave5c.yaml")
    parser.add_argument("--split", default=None)
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    split = args.split or str(cfg["evaluation"]["final_split"])

    selected = json.loads(
        (
            ROOT / cfg["paths"]["output_dir"] / "tuning" / "selected_patient_energy.json"
        ).read_text()
    )
    alpha = float(selected["node_activity_weight"])
    eta = float(selected["gate_weight"])
    variants = energy_variants(
        alpha,
        eta,
        beta_c3=float(cfg["energy"]["static_hcr_prior_ablation"]),
    )

    queries = pd.read_csv(ROOT / cfg["paths"]["output_dir"] / "registry" / "gate_queries.csv")
    pairs = pd.read_csv(
        ROOT / cfg["paths"]["output_dir"] / "registry" / "patient_pair_assignments.csv"
    )
    queries = queries[queries["query_split"] == split]
    pairs = pairs[pairs["query_id"].isin(set(queries["query_id"]))]

    out_dir = ROOT / cfg["paths"]["output_dir"] / "query_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in cfg["seeds"]:
        bundle = load_seed_bundle(cfg, int(seed))
        qseed = queries[queries["seed"] == int(seed)]
        pseed = pairs[pairs["seed"] == int(seed)]
        if qseed.empty:
            continue
        qmap = qseed.set_index("query_id")

        # Patient shuffle mapping for C4 (all base patients in this seed's pairs).
        base_ids = set()
        for _, r in pseed.iterrows():
            if "base_patient_id" in r.index and pd.notna(r.get("base_patient_id")):
                base_ids.add(str(r["base_patient_id"]).split("#")[0])
            else:
                base_ids.add(str(r["patient_id"]).split("#")[0])
        shuffle_map = (
            shuffle_patient_profiles(sorted(base_ids), int(seed)) if base_ids else {}
        )

        # Context shuffle: permute context column labels among contexts in registry.
        ctx_map_c5 = {}
        all_ctx = sorted({c for ctx in bundle["context_map"].values() for c in ctx})
        if all_ctx:
            import numpy as np

            rng = np.random.default_rng(int(seed) + 91)
            perm = rng.permutation(all_ctx)
            remap = dict(zip(all_ctx, [str(x) for x in perm]))
            for edge, ctxs in bundle["context_map"].items():
                ctx_map_c5[edge] = tuple(remap.get(c, c) for c in ctxs)
        else:
            ctx_map_c5 = dict(bundle["context_map"])

        ctx_map_c6 = matched_random_context_map(
            bundle["context_map"], bundle["meta"], int(seed)
        )

        for vname, ecfg in variants.items():
            graph = (
                bundle["graph_hgt"] if vname == "C7_hgt_topology" else bundle["graph_hcr"]
            )
            if vname == "C5_context_shuffle":
                cmap = ctx_map_c5
            elif vname == "C6_matched_random_context":
                cmap = ctx_map_c6
            else:
                cmap = bundle["context_map"]

            for _, asg in pseed.iterrows():
                qid = str(asg["query_id"])
                if qid not in qmap.index:
                    continue
                q = qmap.loc[qid]
                if isinstance(q, pd.DataFrame):
                    q = q.iloc[0]
                query = q.to_dict()
                query["query_id"] = qid

                if vname == "C4_patient_shuffle":
                    activity = resolve_activity(
                        bundle, asg, query, shuffle_map=shuffle_map
                    )
                    pid_out = str(asg["patient_id"])
                else:
                    activity = resolve_activity(bundle, asg, query, shuffle_map=None)
                    pid_out = str(asg["patient_id"])

                metrics = evaluate_patient_query(
                    pid_out,
                    activity,
                    query,
                    graph,
                    bundle["edges"],
                    cmap,
                    bundle["admissible"],
                    bundle["hubs"],
                    ecfg,
                    top_k=int(cfg["evaluation"]["top_k"]),
                )
                rows.append(
                    {
                        "variant": vname,
                        "seed": int(seed),
                        "panel": str(asg["panel"]),
                        "gate_state": str(asg["gate_state"]),
                        "query_split": split,
                        **metrics,
                    }
                )
        print(f"seed={seed} rows={len(rows)}", flush=True)

    df = pd.DataFrame(rows)
    factual = df[df["panel"] == "F"]
    paired = df[df["panel"] == "P"]
    factual.to_csv(out_dir / "factual_query_patient_metrics.csv", index=False)
    paired.to_csv(out_dir / "paired_toggle_query_patient_metrics.csv", index=False)
    print(f"Wrote factual={len(factual)} paired={len(paired)} → {out_dir}")


if __name__ == "__main__":
    main()
