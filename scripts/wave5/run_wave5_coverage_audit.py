#!/usr/bin/env python3
"""Wave 5A.1 — score every frozen query (never drop from the denominator).

TPM_q = P(γ*_q | s_q, e_q) if γ*_q ∈ Ω_{s,e}, else 0.

Writes one row per (variant, seed, query_id) with coverage flags and metrics.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / "src"))

from hcr.motif_registry_v3 import DUAL_GATES
from hcr.motifs import load_truth_graph
from wnerw.finite_path_ensemble import FinitePathEnsemble
from wnerw.gate_factors import (
    apply_gate_potentials_to_edge_weights,
    population_gate_scores_from_d1,
)
from wnerw.graph_builder import build_wnerw_graph
from wnerw.metrics import (
    hit_at_k,
    hub_mass,
    path_entropy,
    reciprocal_rank,
    true_path_rank,
)
from wnerw.potentials import build_edge_log_weights
from wnerw.topk_paths import top_k_paths
from wnerw.types import NEG_INF


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _finite_or_none(value) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(x):
        return None
    return x


def parse_true_path(text: str) -> tuple[str, ...] | None:
    text = str(text or "").strip()
    if not text:
        return None
    nodes = tuple(p.strip() for p in text.split(">") if p.strip())
    return nodes if len(nodes) >= 2 else None


def shuffle_hcr_columns(
    edge_rows: list[dict],
    *,
    mode: str,
    rng: np.random.Generator,
    key: str = "p_hcr2_calibrated",
) -> None:
    """In-place shuffle of HCR potentials among admissible predicted edges."""
    if mode in {None, "", "none"}:
        return
    pred_idx = [i for i, r in enumerate(edge_rows) if not r["edge_in_train"]]
    if not pred_idx:
        return

    if mode in {"edgewise_shuffle_global", "edgewise_global", "global"}:
        vals = [edge_rows[i].get(key) for i in pred_idx]
        rng.shuffle(vals)
        for i, v in zip(pred_idx, vals):
            edge_rows[i][key] = v
            alt = key.replace("_calibrated", "")
            if alt in edge_rows[i]:
                edge_rows[i][alt] = v
        return

    if mode in {"edgewise_shuffle_matched", "edgewise_matched", "matched"}:
        groups: dict[tuple, list[int]] = {}
        for i in pred_idx:
            r = edge_rows[i]
            gkey = (
                str(r.get("edge_type", "unknown")),
                str(r.get("edge_split", "unknown")),
                bool(r.get("hcr_supported", False)),
            )
            groups.setdefault(gkey, []).append(i)
        for idxs in groups.values():
            vals = [edge_rows[i].get(key) for i in idxs]
            rng.shuffle(vals)
            for i, v in zip(idxs, vals):
                edge_rows[i][key] = v
                alt = key.replace("_calibrated", "")
                if alt in edge_rows[i]:
                    edge_rows[i][alt] = v
        return

    raise ValueError(f"Unknown shuffle mode: {mode!r}")


def build_edge_rows(graph: nx.DiGraph, evidence: pd.DataFrame, prob_col: str) -> list[dict]:
    ev_idx = evidence.drop_duplicates(subset=["source", "target"], keep="first").set_index(
        ["source", "target"]
    )
    rows = []
    for u, v in graph.edges:
        key = (u, v)
        if key in ev_idx.index:
            row = ev_idx.loc[key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            rows.append(
                {
                    "source": u,
                    "target": v,
                    "edge_in_train": bool(row.get("edge_in_train", False)),
                    "edge_type": str(row.get("edge_type", "unknown")),
                    "edge_split": str(row.get("edge_split", "unknown")),
                    "hcr_supported": bool(row.get("hcr_supported", False)),
                    "p_hgt": _finite_or_none(row.get("p_hgt_calibrated", row.get("p_hgt")))
                    or 0.5,
                    "p_hgt_calibrated": _finite_or_none(
                        row.get("p_hgt_calibrated", row.get("p_hgt"))
                    )
                    or 0.5,
                    "p_hcr2": _finite_or_none(row.get("p_hcr2_calibrated", row.get("p_hcr2"))),
                    "p_hcr2_calibrated": _finite_or_none(
                        row.get("p_hcr2_calibrated", row.get("p_hcr2"))
                    ),
                    "p_d1": _finite_or_none(row.get("p_d1_calibrated", row.get("p_d1"))),
                    "p_d1_calibrated": _finite_or_none(
                        row.get("p_d1_calibrated", row.get("p_d1"))
                    ),
                    "p_calibrated": _finite_or_none(row.get("p_calibrated", row.get(prob_col)))
                    or 0.5,
                    "p_hcr3": None,
                    "hcr_uncertainty": float(row.get("hcr_uncertainty", 0.0) or 0.0),
                }
            )
        else:
            rows.append(
                {
                    "source": u,
                    "target": v,
                    "edge_in_train": True,
                    "edge_type": "train",
                    "edge_split": "train",
                    "hcr_supported": False,
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
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument(
        "--queries",
        type=Path,
        default=ROOT / "outputs/wave5/queries/wave5_frozen_queries.csv",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", default=None)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Per-query metrics CSV",
    )
    parser.add_argument("--shuffle-seed", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    variant = args.variant or cfg.get("name", args.config.stem)
    evidence = load_table(args.evidence)
    queries = pd.read_csv(args.queries)

    nodes, _ = load_truth_graph(None)
    train_pairs = evidence.loc[
        evidence["edge_in_train"].astype(bool), ["source", "target"]
    ].drop_duplicates()

    prob_col = str(
        cfg.get("graph", {}).get(
            "probability_column",
            cfg.get("edge_potential", {}).get("probability_column", "p_calibrated"),
        )
    )
    evidence = evidence.copy()
    if "p_hgt_calibrated" not in evidence.columns and "p_hgt" in evidence.columns:
        evidence["p_hgt_calibrated"] = evidence["p_hgt"]
    if "p_calibrated" not in evidence.columns and prob_col in evidence.columns:
        evidence["p_calibrated"] = evidence[prob_col]
    if prob_col not in evidence.columns:
        raise KeyError(f"Missing {prob_col!r}")

    graph_src = str(cfg.get("graph", {}).get("source", "predicted"))
    # Predicted candidates only — topological_allowed gates NEW edges, not train.
    if graph_src == "train_only":
        predicted = evidence.iloc[0:0].copy()
        if "p_calibrated" not in predicted.columns:
            predicted["p_calibrated"] = []
    else:
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

    edge_rows = build_edge_rows(graph, evidence, prob_col)
    ep = cfg["edge_potential"]
    shuffle_mode = str(ep.get("shuffle_mode") or "")
    if bool(ep.get("shuffle_hcr", False)) and not shuffle_mode:
        # Backward-compat with Wave 5A shuffled_hcr.yaml
        shuffle_mode = "edgewise_shuffle_global"
    if variant.startswith("shuffled") and not shuffle_mode:
        shuffle_mode = "edgewise_shuffle_global"
    shuffle_key = (
        "p_d1_calibrated"
        if "d1" in str(ep.get("mode", "")).lower() or "d1" in variant
        else "p_hcr2_calibrated"
    )
    if shuffle_mode and shuffle_mode != "none":
        rng = np.random.default_rng(int(args.shuffle_seed or args.seed))
        shuffle_hcr_columns(edge_rows, mode=shuffle_mode, rng=rng, key=shuffle_key)

    weights = build_edge_log_weights(
        edge_rows,
        alpha_hgt=float(ep.get("alpha_hgt", 1.0)),
        beta_hcr2=float(ep.get("beta_hcr2", 1.0)),
        beta_hcr3=float(ep.get("beta_hcr3", 0.0)),
        gamma_delta=float(ep.get("gamma_delta", 1.0)),
        uncertainty_penalty=float(ep.get("uncertainty_penalty", 0.0)),
        length_penalty=float(cfg["path"].get("length_penalty", 0.0)),
        known_edge_probability=float(
            ep.get("known_edge_probability", 1.0 - 1e-6) or 1.0 - 1e-6
        ),
        mode=str(ep.get("mode", "nested_probabilities")),
    )
    motifs_cfg = cfg.get("motifs") or {}
    if bool(motifs_cfg.get("enabled", False)):
        gate_scores = population_gate_scores_from_d1(
            edge_rows,
            coefficient=float(motifs_cfg.get("coefficient", 1.0)),
            probability_key="p_d1_calibrated",
        )
        weights = apply_gate_potentials_to_edge_weights(weights, gate_scores)

    ens = FinitePathEnsemble(
        graph,
        weights,
        temperature=float(cfg["path"].get("temperature", 1.0)),
    )
    k = int(cfg["path"].get("top_k", 20))
    hubs = set(DUAL_GATES)

    rows = []
    for _, q in queries.iterrows():
        qid = str(q["query_id"])
        source, endpoint = str(q["source"]), str(q["endpoint"])
        true_nodes = parse_true_path(q.get("true_path", ""))

        base = {
            "variant": variant,
            "seed": int(args.seed),
            "query_id": qid,
            "source": source,
            "endpoint": endpoint,
            "true_path": q.get("true_path", ""),
            "true_route_family": q.get("true_route_family", ""),
            "hidden_edge": q.get("hidden_edge", ""),
            "mask_split": q.get("mask_split", ""),
        }

        has_any = (
            source in ens.graph
            and endpoint in ens.graph
            and nx.has_path(ens.graph, source, endpoint)
        )
        if not has_any:
            rows.append(
                {
                    **base,
                    "has_any_path": False,
                    "has_true_path": False,
                    "n_paths": 0,
                    "true_path_mass": 0.0,
                    "true_path_rank": float("nan"),
                    "reciprocal_rank": 0.0,
                    "hit_at_1": 0,
                    "hit_at_5": 0,
                    "hit_at_10": 0,
                    "path_entropy": 0.0,
                    "hub_mass": 0.0,
                    "log_partition": float("nan"),
                }
            )
            continue

        log_b = ens.backward_log_partition(endpoint)
        log_z = float(log_b[source])
        if log_z == NEG_INF:
            rows.append(
                {
                    **base,
                    "has_any_path": False,
                    "has_true_path": False,
                    "n_paths": 0,
                    "true_path_mass": 0.0,
                    "true_path_rank": float("nan"),
                    "reciprocal_rank": 0.0,
                    "hit_at_1": 0,
                    "hit_at_5": 0,
                    "hit_at_10": 0,
                    "path_entropy": 0.0,
                    "hub_mass": 0.0,
                    "log_partition": float("nan"),
                }
            )
            continue

        n_paths = ens.count_paths(source, endpoint)
        ranked = top_k_paths(ens, source, endpoint, k=k, log_partition=log_b)
        has_true = bool(true_nodes is not None and ens.path_in_graph(true_nodes))
        if has_true:
            tpm = float(ens.path_probability(true_nodes, log_partition=log_b) or 0.0)
        else:
            tpm = 0.0

        rows.append(
            {
                **base,
                "has_any_path": True,
                "has_true_path": has_true,
                "n_paths": int(n_paths),
                "true_path_mass": tpm,
                "true_path_rank": true_path_rank(ranked, true_nodes),
                "reciprocal_rank": reciprocal_rank(ranked, true_nodes)
                if has_true
                else 0.0,
                "hit_at_1": hit_at_k(ranked, true_nodes, 1) if has_true else 0,
                "hit_at_5": hit_at_k(ranked, true_nodes, 5) if has_true else 0,
                "hit_at_10": hit_at_k(ranked, true_nodes, 10) if has_true else 0,
                "path_entropy": path_entropy([p.probability for p in ranked]),
                "hub_mass": hub_mass(ranked, hubs),
                "log_partition": log_z,
            }
        )

    out = pd.DataFrame(rows)
    # Hard contract: one row per frozen query.
    assert len(out) == len(queries), (len(out), len(queries))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    summary = {
        "variant": variant,
        "seed": int(args.seed),
        "n_queries": int(len(out)),
        "path_coverage": float(out["has_any_path"].mean()),
        "true_path_coverage": float(out["has_true_path"].mean()),
        "unconditional_tpm": float(out["true_path_mass"].mean()),
        "n_graph_nodes": int(graph.number_of_nodes()),
        "n_graph_edges": int(graph.number_of_edges()),
        "n_train_edges_in": int(len(train_pairs)),
        "shuffle_mode": shuffle_mode or "none",
        "out": str(args.out),
    }
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
