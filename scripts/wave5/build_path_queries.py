#!/usr/bin/env python3
"""Build frozen population-level path queries for Wave 5B."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / "src"))

from hcr.motifs import load_truth_graph
from wnerw.graph_builder import LAYER_RANK, layer_rank_from_nodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/wave5/queries/frozen_path_queries.csv",
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
    if "node" not in nodes.columns:
        raise KeyError("nodes missing 'node'")
    ranks = layer_rank_from_nodes(nodes)
    endpoints = [
        str(n)
        for n, r in ranks.items()
        if r == LAYER_RANK["6_endpoints"]
    ]
    # Drug-layer sources (rank 1)
    drugs = [str(n) for n, r in ranks.items() if r == LAYER_RANK["2_drugs"]]

    # Prefer true paths of length ≥ 2 from audited edges for query seeding.
    # Queries themselves do not leak path labels into the scorer.
    edge_set = {
        (str(s), str(t))
        for s, t in edges[["source", "target"]].itertuples(index=False, name=None)
    }

    rows = []
    qid = 0
    for drug in sorted(drugs):
        for endpoint in sorted(endpoints):
            # Keep query if drug can reach endpoint in the audited DAG via BFS on true edges
            # (used only to choose clinically meaningful queries, not for scoring weights).
            import networkx as nx

            g = nx.DiGraph()
            g.add_edges_from(edge_set)
            if drug not in g or endpoint not in g:
                continue
            if not nx.has_path(g, drug, endpoint):
                continue
            rows.append(
                {
                    "query_id": f"q{qid:04d}",
                    "source": drug,
                    "endpoint": endpoint,
                    "patient_id": None,
                    "level": "population",
                }
            )
            qid += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(rows)} queries)")


if __name__ == "__main__":
    main()
