#!/usr/bin/env python3
"""Build the frozen Wave-5 query registry (shared by all variants/seeds).

Writes:
  outputs/wave5/queries/wave5_frozen_queries.csv

Columns:
  query_id, source, endpoint, true_path, true_route_family, hidden_edge, mask_split

G_true is used only here (registry construction / evaluator oracle labels).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import pandas as pd
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / "src"))

from hcr.motif_registry_v3 import DUAL_GATES, TRIPLE_GATES
from hcr.motifs import load_truth_graph
from wnerw.evaluation import true_paths_between, truth_digraph_from_edges
from wnerw.graph_builder import LAYER_RANK, layer_rank_from_nodes


def route_family(path: tuple[str, ...]) -> str:
    hubs = set(DUAL_GATES) | set(TRIPLE_GATES)
    hit = [n for n in path if n in hubs]
    if hit:
        return f"via_gate:{hit[0]}"
    return "mechanism_chain"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-queries",
        type=Path,
        default=ROOT / "outputs/wave5/queries/frozen_path_queries.csv",
        help="Existing source/endpoint list to freeze (keeps query_ids stable).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/wave5/queries/wave5_frozen_queries.csv",
    )
    parser.add_argument(
        "--mask-split",
        default="none",
        help="Population 5A has no motif hide; use none.",
    )
    args = parser.parse_args()

    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = compose(
            config_name="config",
            overrides=[
                "model=TaskA_hgt",
                "hcr=none",
                "data.dataset.scenario=clean",
                "wandb.enabled=false",
            ],
        )
    nodes, edges = load_truth_graph(cfg)
    g_true = truth_digraph_from_edges(edges)
    ranks = layer_rank_from_nodes(nodes)

    if args.seed_queries.exists():
        base = pd.read_csv(args.seed_queries)
    else:
        # Fallback: rebuild drug→endpoint reachable queries.
        endpoints = [n for n, r in ranks.items() if r == LAYER_RANK["6_endpoints"]]
        drugs = [n for n, r in ranks.items() if r == LAYER_RANK["2_drugs"]]
        rows = []
        qid = 0
        for drug in sorted(drugs):
            for endpoint in sorted(endpoints):
                if drug in g_true and endpoint in g_true and nx.has_path(
                    g_true, drug, endpoint
                ):
                    rows.append(
                        {
                            "query_id": f"q{qid:04d}",
                            "source": drug,
                            "endpoint": endpoint,
                        }
                    )
                    qid += 1
        base = pd.DataFrame(rows)

    out_rows = []
    for _, q in base.iterrows():
        qid = str(q["query_id"])
        source, endpoint = str(q["source"]), str(q["endpoint"])
        paths = true_paths_between(g_true, source, endpoint, max_paths=50, cutoff=8)
        if not paths:
            # Keep the query anyway — TPM will be 0 everywhere.
            true_path = ""
            family = "unreachable_in_true"
        else:
            # Designate shortest true path (stable tie-break lexicographic).
            paths_sorted = sorted(paths, key=lambda p: (len(p), p))
            true_path_nodes = paths_sorted[0]
            true_path = " > ".join(true_path_nodes)
            family = route_family(true_path_nodes)
        out_rows.append(
            {
                "query_id": qid,
                "source": source,
                "endpoint": endpoint,
                "true_path": true_path,
                "true_route_family": family,
                "hidden_edge": "",
                "mask_split": str(args.mask_split),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(out_rows)
    df.to_csv(args.out, index=False)
    n_ok = int((df["true_path"].astype(str).str.len() > 0).sum())
    print(f"Wrote {args.out} ({len(df)} queries, {n_ok} with designated true_path)")


if __name__ == "__main__":
    main()
