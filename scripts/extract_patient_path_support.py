#!/usr/bin/env python3
"""Extract static + patient path/gate features for Wave 5D (per seed / control)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from link_prediction.wave5d.patient_edge_features import (
    build_edge_feature_bundle,
    save_feature_bundle,
)
from wnerw.wave5c.edge_context_registry import (
    admissible_patient_feature_table,
    build_context_map,
    build_edge_context_registry,
    g_train_from_evidence_and_hetero,
)
from wnerw.wave5c.patient_energy import PatientEnergyConfig
from wnerw.wave5c.runtime import compose_task_cfg
from hcr.structural_context import load_node_metadata


CONTROLS = {
    "none": "none",
    "L5_patient_shuffle": "patient_shuffle",
    "L6_context_shuffle": "context_shuffle",
    "L7_matched_random_context": "matched_random_context",
    "L8_random_path_weights": "random_path_weights",
    "L10_no_leave_one_out": "no_leave_one_out",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/link_prediction/wave5d_clean.yaml",
    )
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--control",
        type=str,
        default="none",
        choices=sorted(set(CONTROLS.values()) | {"none"}),
    )
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    seeds = [args.seed] if args.seed is not None else list(cfg["seeds"])
    out_dir = ROOT / cfg["paths"]["output_dir"] / "patient_features"
    out_dir.mkdir(parents=True, exist_ok=True)

    ps = cfg["path_support"]
    energy = PatientEnergyConfig(
        temperature=float(ps["temperature"]),
        length_penalty=float(ps["length_penalty"]),
        node_activity_weight=float(ps.get("node_activity_weight", 0.0)),
        gate_weight=float(ps["gate_weight"]),
        static_hcr_weight=float(ps.get("static_hcr_weight", 0.0)),
    )
    max_p = int(ps.get("max_patients_per_split", 48))

    # Prefer Wave 5C registry contexts when available
    reg_path = ROOT / cfg["paths"]["wave5c_registry"]
    registry_all = pd.read_csv(reg_path) if reg_path.exists() else None

    for seed in seeds:
        edge_path = ROOT / cfg["paths"]["evidence_pattern"].format(seed=seed)
        evidence = pd.read_csv(edge_path)
        cand_path = (
            ROOT
            / cfg["paths"]["output_dir"]
            / "candidates"
            / f"frozen_candidates_seed{seed}.csv"
        )
        if not cand_path.exists():
            cand_path = ROOT / cfg["paths"]["output_dir"] / "candidates" / "frozen_candidates.csv"
        candidates = pd.read_csv(cand_path)
        if "seed" in candidates.columns:
            candidates = candidates[candidates["seed"] == seed].drop(columns=["seed"])

        task_cfg = compose_task_cfg(seed)
        from data.PreprocessingTaskA.load_hetero_recon_data import load_recon_heterodata
        from data.patient_matrix import load_patient_matrix_with_split

        train, _, _, _ = load_recon_heterodata(task_cfg)
        patients = load_patient_matrix_with_split(task_cfg)
        g_train = g_train_from_evidence_and_hetero(train)
        meta = load_node_metadata(task_cfg)
        adm_df = admissible_patient_feature_table(meta)
        admissible = set(adm_df.loc[adm_df["admissible"], "node"].astype(str))

        if registry_all is not None and "seed" in registry_all.columns:
            reg = registry_all[registry_all["seed"] == seed]
        elif registry_all is not None:
            reg = registry_all
        else:
            reg = build_edge_context_registry(
                g_train, evidence, meta, seed=seed, working_graph=g_train
            )
        context_map = build_context_map(reg)

        print(
            f"seed={seed} control={args.control} "
            f"candidates={len(candidates)} contexts={len(context_map)}"
        )
        bundle = build_edge_feature_bundle(
            cfg=task_cfg,
            train_data=train,
            evidence=evidence,
            candidates=candidates,
            patients=patients,
            context_map=context_map,
            admissible=admissible,
            g_train=g_train,
            seed=int(seed),
            root=ROOT,
            energy_cfg=energy,
            max_patients_per_split=max_p,
            device=torch.device("cpu"),
            control=args.control,
        )
        tag = args.control if args.control != "none" else "base"
        out = out_dir / f"features_seed{seed}_{tag}.npz"
        save_feature_bundle(bundle, out)
        meta_out = {
            "seed": int(seed),
            "control": args.control,
            "n_edges": len(bundle.candidate_ids),
            "n_patients": len(bundle.patient_ids),
            "hgt_pair_dim": int(bundle.hgt_pair.shape[1]),
            "path": str(out),
        }
        (out_dir / f"features_seed{seed}_{tag}.json").write_text(
            json.dumps(meta_out, indent=2)
        )
        print(f"  wrote {out} patients={meta_out['n_patients']}")


if __name__ == "__main__":
    main()
