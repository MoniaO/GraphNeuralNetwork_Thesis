#!/usr/bin/env python3
"""Run population-level WNERW (Wave 5A finite-path engine).

Expects evidence tables from ``export_edge_evidence.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / "src"))

from hcr.motifs import load_truth_graph
from wnerw.finite_path_ensemble import FinitePathEnsemble
from wnerw.gate_factors import (
    apply_gate_potentials_to_edge_weights,
    population_gate_scores_from_d1,
)
from wnerw.graph_builder import build_wnerw_graph
from wnerw.metrics import path_entropy, path_hhi
from wnerw.path_marginals import edge_marginals, node_marginals
from wnerw.potentials import build_edge_log_weights
from wnerw.topk_paths import top_k_paths


def load_evidence(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _finite_or_none(value) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/wnerw/hcr2_finite.yaml",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/wave5/paths",
    )
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--shuffle-seed", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    variant = args.variant or cfg.get("name", args.config.stem)
    evidence = load_evidence(args.evidence)
    queries = pd.read_csv(args.queries)
    if args.max_queries is not None:
        queries = queries.head(int(args.max_queries))

    nodes, _ = load_truth_graph(None)
    # Train edges are always passed to the builder (never gated by topological_allowed).
    train_pairs = evidence.loc[
        evidence["edge_in_train"].astype(bool), ["source", "target"]
    ].drop_duplicates()

    graph_src = str(cfg.get("graph", {}).get("source", "predicted"))
    prob_col = str(
        cfg.get("graph", {}).get(
            "probability_column",
            cfg.get("edge_potential", {}).get("probability_column", "p_calibrated"),
        )
    )
    evidence = evidence.copy()
    if "p_hgt_calibrated" not in evidence.columns and "p_hgt" in evidence.columns:
        evidence["p_hgt_calibrated"] = evidence["p_hgt"]
    if "p_d1_calibrated" not in evidence.columns and "p_d1" in evidence.columns:
        evidence["p_d1_calibrated"] = evidence["p_d1"]
    if "p_calibrated" not in evidence.columns:
        if prob_col in evidence.columns:
            evidence["p_calibrated"] = evidence[prob_col]
        elif "p_d1_calibrated" in evidence.columns:
            evidence["p_calibrated"] = evidence["p_d1_calibrated"]
    if prob_col not in evidence.columns:
        raise KeyError(
            f"probability_column {prob_col!r} missing from evidence "
            f"(have {list(evidence.columns)})"
        )

    if graph_src == "train_only":
        predicted = evidence.iloc[0:0].copy()
    else:
        # topological_allowed applies ONLY to new predicted candidates.
        predicted = evidence.loc[
            ~evidence["edge_in_train"].astype(bool)
            & evidence["topological_allowed"].astype(bool),
            ["source", "target", prob_col],
        ].drop_duplicates(subset=["source", "target"])
        if prob_col != "p_calibrated":
            predicted = predicted.rename(columns={prob_col: "p_calibrated"})

    thr = cfg["graph"].get("candidate_threshold")
    threshold = 0.0 if thr is None else float(thr)

    graph = build_wnerw_graph(
        train_pairs,
        predicted,
        nodes,
        threshold=threshold,
        candidate_budget=cfg["graph"].get("candidate_budget"),
    )

    edge_rows = []
    ev_idx = evidence.drop_duplicates(subset=["source", "target"], keep="first").set_index(
        ["source", "target"]
    )
    for u, v in graph.edges:
        key = (u, v)
        if key in ev_idx.index:
            row = ev_idx.loc[key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            edge_rows.append(
                {
                    "source": u,
                    "target": v,
                    "edge_in_train": bool(row.get("edge_in_train", False)),
                    "p_hgt": _finite_or_none(row.get("p_hgt_calibrated", row.get("p_hgt")))
                    or 0.5,
                    "p_hgt_calibrated": _finite_or_none(
                        row.get("p_hgt_calibrated", row.get("p_hgt"))
                    )
                    or 0.5,
                    "p_hcr2": _finite_or_none(
                        row.get("p_hcr2_calibrated", row.get("p_hcr2"))
                    ),
                    "p_hcr2_calibrated": _finite_or_none(
                        row.get("p_hcr2_calibrated", row.get("p_hcr2"))
                    ),
                    "p_d1": _finite_or_none(row.get("p_d1_calibrated", row.get("p_d1"))),
                    "p_d1_calibrated": _finite_or_none(
                        row.get("p_d1_calibrated", row.get("p_d1"))
                    ),
                    "p_calibrated": _finite_or_none(
                        row.get("p_calibrated", row.get(prob_col))
                    )
                    or 0.5,
                    "p_hcr3": None,
                    "hcr_uncertainty": float(row.get("hcr_uncertainty", 0.0) or 0.0),
                }
            )
        else:
            edge_rows.append(
                {
                    "source": u,
                    "target": v,
                    "edge_in_train": True,
                    "p_hgt": 0.999999,
                    "p_hgt_calibrated": 0.999999,
                    "p_hcr2": None,
                    "p_hcr2_calibrated": None,
                    "p_d1": None,
                    "p_d1_calibrated": None,
                    "p_calibrated": 0.999999,
                    "p_hcr3": None,
                    "hcr_uncertainty": 0.0,
                }
            )

    # Control: shuffle HCR-2 / D1 potentials among predicted edges.
    ep = cfg["edge_potential"]
    shuffle_hcr = bool(ep.get("shuffle_hcr", False)) or variant in {
        "shuffled_hcr",
        "shuffled_d1",
    }
    shuffle_key = "p_d1_calibrated" if "d1" in str(ep.get("mode", "")).lower() or "d1" in variant else "p_hcr2"
    if shuffle_hcr:
        rng = np.random.default_rng(int(args.shuffle_seed or args.seed))
        pred_idx = [i for i, r in enumerate(edge_rows) if not r["edge_in_train"]]
        if shuffle_key == "p_d1_calibrated":
            vals = [edge_rows[i]["p_d1_calibrated"] for i in pred_idx]
            rng.shuffle(vals)
            for i, v in zip(pred_idx, vals):
                edge_rows[i]["p_d1_calibrated"] = v
                edge_rows[i]["p_d1"] = v
        else:
            vals = [edge_rows[i]["p_hcr2"] for i in pred_idx]
            rng.shuffle(vals)
            for i, v in zip(pred_idx, vals):
                edge_rows[i]["p_hcr2"] = v
                edge_rows[i]["p_hcr2_calibrated"] = v

    weights = build_edge_log_weights(
        edge_rows,
        alpha_hgt=float(ep.get("alpha_hgt", 1.0)),
        beta_hcr2=float(ep.get("beta_hcr2", 1.0)),
        beta_hcr3=float(ep.get("beta_hcr3", 0.0)),
        gamma_delta=float(ep.get("gamma_delta", 1.0)),
        uncertainty_penalty=float(ep.get("uncertainty_penalty", 0.0)),
        length_penalty=float(cfg["path"].get("length_penalty", 0.0)),
        known_edge_probability=float(ep.get("known_edge_probability", 1.0 - 1e-6) or 1.0 - 1e-6),
        mode=str(ep.get("mode", "nested_probabilities")),
    )

    # Wave 5B — population gate factors (added once on edges entering the gate).
    motifs_cfg = cfg.get("motifs") or {}
    n_gates = 0
    if bool(motifs_cfg.get("enabled", False)):
        gate_scores = population_gate_scores_from_d1(
            edge_rows,
            coefficient=float(motifs_cfg.get("coefficient", 1.0)),
            probability_key="p_d1_calibrated",
        )
        weights = apply_gate_potentials_to_edge_weights(weights, gate_scores)
        n_gates = len(gate_scores)

    ens = FinitePathEnsemble(
        graph,
        weights,
        temperature=float(cfg["path"].get("temperature", 1.0)),
    )
    k = int(cfg["path"].get("top_k", 20))

    path_rows = []
    marginal_rows = []
    query_rows = []
    for _, q in queries.iterrows():
        source, endpoint = str(q["source"]), str(q["endpoint"])
        if source not in graph or endpoint not in graph:
            continue
        if not nx.has_path(graph, source, endpoint):
            continue
        log_b = ens.backward_log_partition(endpoint)
        log_z = float(log_b[source])
        ranked = top_k_paths(ens, source, endpoint, k=k, log_partition=log_b)
        ent = path_entropy([p.probability for p in ranked])
        hhi = path_hhi([p.probability for p in ranked])
        query_rows.append(
            {
                "seed": args.seed,
                "variant": variant,
                "query_id": q["query_id"],
                "source": source,
                "endpoint": endpoint,
                "log_partition": log_z,
                "n_top_paths": len(ranked),
                "path_entropy": ent,
                "path_hhi": hhi,
            }
        )
        for rank, p in enumerate(ranked, start=1):
            path_rows.append(
                {
                    "seed": args.seed,
                    "variant": variant,
                    "query_id": q["query_id"],
                    "source": source,
                    "endpoint": endpoint,
                    "path_rank": rank,
                    "path_nodes": " > ".join(p.nodes),
                    "path_length": len(p.nodes) - 1,
                    "log_weight": p.log_weight,
                    "path_probability": p.probability,
                    "path_entropy": ent,
                    "path_hhi": hhi,
                    "log_partition": log_z,
                }
            )
        e_marg = edge_marginals(ens, source, endpoint, log_backward=log_b)
        # Keep top-20 edge marginals per query for readability.
        top_e = sorted(e_marg.items(), key=lambda kv: kv[1], reverse=True)[:20]
        for (u, v), mass in top_e:
            marginal_rows.append(
                {
                    "seed": args.seed,
                    "variant": variant,
                    "query_id": q["query_id"],
                    "source": source,
                    "endpoint": endpoint,
                    "edge_source": u,
                    "edge_target": v,
                    "edge_marginal": mass,
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"top_paths_seed{args.seed}_{variant}.csv"
    df = pd.DataFrame(path_rows)
    df.to_csv(out, index=False)
    pd.DataFrame(query_rows).to_csv(
        args.out_dir / f"queries_seed{args.seed}_{variant}.csv", index=False
    )
    pd.DataFrame(marginal_rows).to_csv(
        args.out_dir / f"edge_marginals_seed{args.seed}_{variant}.csv", index=False
    )
    summary = {
        "seed": args.seed,
        "variant": variant,
        "n_paths": int(len(df)),
        "n_queries_scored": int(df["query_id"].nunique()) if len(df) else 0,
        "n_graph_nodes": int(graph.number_of_nodes()),
        "n_graph_edges": int(graph.number_of_edges()),
        "n_gate_factors": int(n_gates),
        "edge_potential_mode": str(ep.get("mode", "nested_probabilities")),
        "graph_source": graph_src,
        "out": str(out),
    }
    (args.out_dir / f"summary_seed{args.seed}_{variant}.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
