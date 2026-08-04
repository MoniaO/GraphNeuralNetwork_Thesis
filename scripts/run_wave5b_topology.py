#!/usr/bin/env python3
"""Wave 5B Panel T — pure topology (q_uv = 0 on each model's own graph)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcr.motifs import load_truth_graph
from wnerw.wave5b.energy_maps import uniform_edge_scores
from wnerw.wave5b.graph_builders import (
    build_model_graph,
    build_train_only_graph,
    hub_nodes_from_degree,
)
from wnerw.wave5b.io import load_edges, load_queries, merge_patient_shuffle, resolve_path
from wnerw.wave5b.paired_summary import summarize_panel_t
from wnerw.wave5b.query_evaluator import (
    evaluate_query,
    parse_hidden_edge,
    parse_true_path,
)


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/wnerw/wave5b.yaml",
    )
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    out_dir = ROOT / cfg["paths"]["output_dir"] / "panel_t"
    out_dir.mkdir(parents=True, exist_ok=True)

    queries = load_queries(ROOT / cfg["paths"]["query_registry"])
    nodes, _ = load_truth_graph(None)
    thr = cfg["graph_selection"]
    cols = cfg["columns"]
    top_k = int(cfg["evaluation"]["top_k"])
    q_hub = float(cfg["hub_definition"]["degree_quantile"])

    rows = []
    for seed in cfg["seeds"]:
        edge_path = resolve_path(cfg["paths"]["edge_pattern"], seed, ROOT)
        ps_path = resolve_path(cfg["paths"]["patient_shuffle_pattern"], seed, ROOT)
        edges = load_edges(edge_path)
        edges = merge_patient_shuffle(
            edges,
            ps_path,
            out_col=cols["patient_shuffle_probability"],
        )

        graphs = {
            "T0_train": build_train_only_graph(edges, nodes=nodes),
            "T1_hgt": build_model_graph(
                edges,
                cols["hgt_probability"],
                float(thr["hgt_threshold"]),
                nodes=nodes,
            ),
            "T2_hcr": build_model_graph(
                edges,
                cols["hcr_probability"],
                float(thr["hcr_threshold"]),
                nodes=nodes,
            ),
            "T3_patient_shuffle": build_model_graph(
                edges,
                cols["patient_shuffle_probability"],
                float(thr["patient_shuffle_threshold"]),
                nodes=nodes,
            ),
        }

        for variant, graph in graphs.items():
            hubs = hub_nodes_from_degree(graph, degree_quantile=q_hub)
            scores = uniform_edge_scores(graph)
            for _, q in queries.iterrows():
                true_path = parse_true_path(q["true_path"])
                hidden = parse_hidden_edge(q.get("hidden_edge"), true_path)
                metrics = evaluate_query(
                    query_id=str(q["query_id"]),
                    source=str(q["source"]),
                    endpoint=str(q["endpoint"]),
                    true_path=true_path,
                    hidden_edge=hidden,
                    graph=graph,
                    edge_scores=scores,
                    hubs=hubs,
                    top_k=top_k,
                    compute_n_paths=True,
                )
                rows.append(
                    {
                        "variant": variant,
                        "seed": int(seed),
                        "query_split": str(q.get("query_split", "")),
                        **metrics,
                    }
                )
        print(f"Panel T seed={seed} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "query_metrics.csv", index=False)
    summary = summarize_panel_t(df)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
