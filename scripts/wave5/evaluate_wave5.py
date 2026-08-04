#!/usr/bin/env python3
"""Evaluate Wave 5 path outputs against G_true paths (evaluator only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / "src"))

from hcr.motifs import load_truth_graph
from hcr.motif_registry_v3 import DUAL_GATES
from wnerw.evaluation import (
    score_query_against_truth,
    true_paths_between,
    truth_digraph_from_edges,
)
from wnerw.topk_paths import RankedPath


def load_paths(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    df = load_paths(args.paths)
    _, edges = load_truth_graph(None)
    g_true = truth_digraph_from_edges(edges)
    hubs = set(DUAL_GATES)

    rows = []
    for (qid, seed, variant), sub in df.groupby(
        ["query_id", "seed", "variant"], sort=False
    ):
        sub = sub.sort_values("path_rank")
        source = str(sub.iloc[0]["source"])
        endpoint = str(sub.iloc[0]["endpoint"])
        ranked = [
            RankedPath(
                nodes=tuple(str(x).strip() for x in str(r["path_nodes"]).split(">")),
                log_weight=float(r["log_weight"]),
                probability=float(r["path_probability"]),
            )
            for _, r in sub.iterrows()
        ]
        # Fix split artifacts: "a > b" → strip already done
        ranked = [
            RankedPath(
                nodes=tuple(n.strip() for n in p.nodes),
                log_weight=p.log_weight,
                probability=p.probability,
            )
            for p in ranked
        ]
        true = true_paths_between(g_true, source, endpoint)
        metrics = score_query_against_truth(ranked, true, hub_nodes=hubs)
        rows.append(
            {
                "query_id": qid,
                "seed": int(seed),
                "variant": variant,
                "source": source,
                "endpoint": endpoint,
                **metrics,
            }
        )

    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    summary = {
        "n_queries": int(len(out)),
        "mean_true_path_mass": float(out["true_path_mass"].mean()) if len(out) else None,
        "mean_path_entropy": float(out["path_entropy"].mean()) if len(out) else None,
        "mean_recall_at_5": float(out["path_recall_at_5"].mean()) if len(out) else None,
        "out": str(args.out),
    }
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
