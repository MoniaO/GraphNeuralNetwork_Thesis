#!/usr/bin/env python3
"""Build gate queries + factual / paired-toggle patient assignments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wnerw.wave5c.patient_activity import (
    build_patient_activity,
    gate_activation,
    make_gate_toggle_profiles,
    make_triple_off_profiles,
)
from wnerw.wave5c.patient_queries import build_gate_queries, parse_context_nodes
from wnerw.wave5c.runtime import load_cfg, load_seed_bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/wnerw/wave5c.yaml")
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    out = ROOT / cfg["paths"]["output_dir"] / "registry"
    out.mkdir(parents=True, exist_ok=True)
    max_n = int(cfg["evaluation"]["max_patients_per_query_state"])

    query_frames = []
    pair_frames = []

    for seed in cfg["seeds"]:
        bundle = load_seed_bundle(cfg, int(seed))
        queries = build_gate_queries(bundle["graph_hcr"], bundle["context_map"])
        if len(queries) == 0:
            print(f"seed={seed}: no gate queries", flush=True)
            continue
        queries["seed"] = int(seed)
        query_frames.append(queries)

        patients = bundle["patients"]
        # Prefer validation/test patients for evaluation; train for prevalence only.
        eval_patients = patients[patients["split"].isin(["valid", "validation", "test"])]
        if eval_patients.empty:
            eval_patients = patients

        for _, q in queries.iterrows():
            src = str(q["source"])
            ctx = parse_context_nodes(q["context_nodes"])
            # Factual ON / OFF among source-active patients
            on_ids, off_ids = [], []
            for _, prow in eval_patients.iterrows():
                act = build_patient_activity(prow, bundle["admissible"] | {src} | set(ctx))
                if act.get(src, 0.0) < 0.5:
                    continue
                gact = gate_activation(src, ctx, act)
                pid = str(prow["patient_id"])
                if gact >= 0.5:
                    on_ids.append(pid)
                else:
                    off_ids.append(pid)
            on_ids = on_ids[:max_n]
            off_ids = off_ids[:max_n]
            for pid in on_ids:
                pair_frames.append(
                    {
                        "seed": int(seed),
                        "query_id": q["query_id"],
                        "panel": "F",
                        "gate_state": "ON",
                        "patient_id": pid,
                        "profile_mode": "factual",
                    }
                )
            for pid in off_ids:
                pair_frames.append(
                    {
                        "seed": int(seed),
                        "query_id": q["query_id"],
                        "panel": "F",
                        "gate_state": "OFF",
                        "patient_id": pid,
                        "profile_mode": "factual",
                    }
                )

            # Paired toggles: base on source-active patients (up to max_n)
            bases = (on_ids + off_ids)[:max_n]
            for pid in bases:
                prow = eval_patients.loc[
                    eval_patients["patient_id"].astype(str) == pid
                ].iloc[0]
                act = build_patient_activity(prow, bundle["admissible"] | {src} | set(ctx))
                if len(ctx) >= 2:
                    ons = []
                    offs = make_triple_off_profiles(act, src, ctx)
                    on_prof, _ = make_gate_toggle_profiles(act, src, ctx)
                    ons.append(on_prof)
                    # one ON + mean over OFF variants later in runner
                    pair_frames.append(
                        {
                            "seed": int(seed),
                            "query_id": q["query_id"],
                            "panel": "P",
                            "gate_state": "ON",
                            "patient_id": pid,
                            "profile_mode": "toggle",
                            "profile_json": json.dumps(on_prof),
                        }
                    )
                    for j, off_prof in enumerate(offs):
                        pair_frames.append(
                            {
                                "seed": int(seed),
                                "query_id": q["query_id"],
                                "panel": "P",
                                "gate_state": "OFF",
                                "patient_id": f"{pid}#off{j}",
                                "base_patient_id": pid,
                                "profile_mode": "toggle",
                                "profile_json": json.dumps(off_prof),
                            }
                        )
                else:
                    on_prof, off_prof = make_gate_toggle_profiles(act, src, ctx)
                    pair_frames.append(
                        {
                            "seed": int(seed),
                            "query_id": q["query_id"],
                            "panel": "P",
                            "gate_state": "ON",
                            "patient_id": pid,
                            "profile_mode": "toggle",
                            "profile_json": json.dumps(on_prof),
                        }
                    )
                    pair_frames.append(
                        {
                            "seed": int(seed),
                            "query_id": q["query_id"],
                            "panel": "P",
                            "gate_state": "OFF",
                            "patient_id": pid,
                            "profile_mode": "toggle",
                            "profile_json": json.dumps(off_prof),
                        }
                    )

        print(
            f"seed={seed} queries={len(queries)} pair_rows_partial={len(pair_frames)}",
            flush=True,
        )

    if query_frames:
        qdf = pd.concat(query_frames, ignore_index=True)
        qdf.to_csv(out / "gate_queries.csv", index=False)
    if pair_frames:
        pdf = pd.DataFrame(pair_frames)
        pdf.to_csv(out / "patient_pair_assignments.csv", index=False)
        print(f"Wrote pairs {len(pdf)} → {out}")


if __name__ == "__main__":
    main()
