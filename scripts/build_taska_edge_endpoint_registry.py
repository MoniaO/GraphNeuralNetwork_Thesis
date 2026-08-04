#!/usr/bin/env python3
"""Build frozen node→endpoint reachability + edge registry (seed 20260722)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_GSN = Path.home() / "Desktop" / "GSN Graphs dysertation 2026"
OUT = ROOT / "outputs" / "wave11_taskA" / "mlp_vs_kan_full"


def main() -> None:
    from evaluation.taskA_endpoint_reachability import (
        DEFAULT_ENDPOINTS,
        build_node_endpoint_table,
        expand_candidates_to_endpoint_registry,
        load_audited_digraph,
    )
    from data.PreprocessingTaskA.load_hetero_recon_data import make_candidates

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260722)
    args = ap.parse_args()

    gsn = Path(os.environ.get("GSN_PROJECT_ROOT", DEFAULT_GSN))
    data_dir = gsn / "2 v3. Data" / "dataset_v3"
    edges = data_dir / "synthetic_pharmacotherapy_v3_edges_audited.csv"
    nodes = data_dir / "synthetic_pharmacotherapy_v3_nodes.csv"

    graph = load_audited_digraph(edges, nodes_csv=nodes)
    node_ep = build_node_endpoint_table(graph, DEFAULT_ENDPOINTS)
    OUT.mkdir(parents=True, exist_ok=True)
    node_path = OUT / "node_endpoint_reachability.csv"
    node_ep.to_csv(node_path, index=False)

    cands = make_candidates(
        data_dir,
        negative_ratio=3,
        seed=int(args.seed),
        nodes_file="synthetic_pharmacotherapy_v3_nodes.csv",
        edges_file="synthetic_pharmacotherapy_v3_edges_audited.csv",
        candidate_id_prefix="v3_c",
    )
    # make_candidates returns full table before split — may already have split.
    # If not, attach a dummy split for registry completeness.
    if "split" not in cands.columns:
        cands = cands.copy()
        cands["split"] = "unsplit"
    reg = expand_candidates_to_endpoint_registry(cands, node_ep)
    reg_path = OUT / "edge_endpoint_registry.csv"
    reg.to_csv(reg_path, index=False)
    print(f"wrote {node_path} rows={len(node_ep)}")
    print(f"wrote {reg_path} rows={len(reg)}")
    print(
        "reachable_frac=",
        float(reg["is_downstream_reachable"].mean()),
        "endpoints=",
        list(DEFAULT_ENDPOINTS),
    )


if __name__ == "__main__":
    main()
