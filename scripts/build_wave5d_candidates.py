#!/usr/bin/env python3
"""Build frozen Wave 5D candidate registry from Wave-5 evidence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from link_prediction.wave5d.candidate_registry import (
    build_frozen_candidates,
    save_frozen_candidates,
)
from wnerw.wave5c.edge_context_registry import g_train_from_evidence_and_hetero
from wnerw.wave5c.runtime import compose_task_cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/link_prediction/wave5d_clean.yaml",
    )
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out_dir = ROOT / cfg["paths"]["output_dir"] / "candidates"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for seed in cfg["seeds"]:
        edge_path = ROOT / cfg["paths"]["evidence_pattern"].format(seed=seed)
        evidence = pd.read_csv(edge_path)
        task_cfg = compose_task_cfg(seed)
        from data.PreprocessingTaskA.load_hetero_recon_data import load_recon_heterodata
        from hcr.motifs import load_truth_graph

        train, _, _, _ = load_recon_heterodata(task_cfg)
        g_train = g_train_from_evidence_and_hetero(train)
        nodes, _ = load_truth_graph(task_cfg)
        node_meta = nodes.rename(columns={"node": "node_id"}) if "node" in nodes.columns else nodes
        cand = build_frozen_candidates(
            evidence,
            g_train=g_train,
            node_metadata=node_meta,
            seed=int(seed),
        )
        save_frozen_candidates(cand, out_dir / f"frozen_candidates_seed{seed}.csv")
        frames.append(cand)
        print(f"seed={seed} candidates={len(cand)}")

    all_df = pd.concat(frames, ignore_index=True)
    # Canonical freeze: seed 20260722 (candidate_seed)
    ref = all_df[all_df["seed"] == 20260722].drop(columns=["seed"])
    save_frozen_candidates(ref, out_dir / "frozen_candidates.csv")
    print(f"Wrote {out_dir / 'frozen_candidates.csv'} rows={len(ref)}")


if __name__ == "__main__":
    main()
