#!/usr/bin/env python3
"""Wave 5B — tune (T, λ_L) on validation queries for HCR energy on G_fixed."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcr.motifs import load_truth_graph
from wnerw.wave5b.energy_maps import EnergyConfig, build_edge_energies
from wnerw.wave5b.graph_builders import (
    SelectionSpec,
    build_fixed_union_graph,
    hub_nodes_from_degree,
)
from wnerw.wave5b.io import load_edges, load_queries, merge_patient_shuffle, resolve_path
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
    out_dir = ROOT / cfg["paths"]["output_dir"] / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)

    queries_all = load_queries(ROOT / cfg["paths"]["query_registry"])
    tune_split = str(cfg["evaluation"]["tune_split"])
    queries = queries_all[queries_all["query_split"] == tune_split].copy()
    nodes, _ = load_truth_graph(None)
    thr = cfg["graph_selection"]
    cols = cfg["columns"]
    top_k = int(cfg["evaluation"]["top_k"])
    q_hub = float(cfg["hub_definition"]["degree_quantile"])

    temps = list(cfg["energy"]["temperatures"])
    lengths = list(cfg["energy"]["length_penalties"])
    grid = list(product(temps, lengths))

    # Aggregate validation metrics across seeds for each (T, λ_L).
    grid_rows = []
    for temperature, length_penalty in grid:
        mrr_hcr, mrr_hgt, tpm_hcr, ent_top1, hub_hcr, hub_hgt = [], [], [], [], [], []
        n_eval = 0
        for seed in cfg["seeds"]:
            edge_path = resolve_path(cfg["paths"]["edge_pattern"], seed, ROOT)
            ps_path = resolve_path(cfg["paths"]["patient_shuffle_pattern"], seed, ROOT)
            edges = merge_patient_shuffle(
                load_edges(edge_path),
                ps_path,
                out_col=cols["patient_shuffle_probability"],
            )
            specs = [
                SelectionSpec("hgt", cols["hgt_probability"], float(thr["hgt_threshold"])),
                SelectionSpec("hcr", cols["hcr_probability"], float(thr["hcr_threshold"])),
                SelectionSpec(
                    "ps",
                    cols["patient_shuffle_probability"],
                    float(thr["patient_shuffle_threshold"]),
                ),
            ]
            graph = build_fixed_union_graph(edges, specs, nodes=nodes)
            hubs = hub_nodes_from_degree(graph, degree_quantile=q_hub)

            cfg_hcr = EnergyConfig(
                probability_column=cols["hcr_probability"],
                temperature=float(temperature),
                length_penalty=float(length_penalty),
                uncertainty_penalty=0.0,
                uncertainty_column=cols["uncertainty"],
            )
            cfg_hgt = EnergyConfig(
                probability_column=cols["hgt_probability"],
                temperature=float(temperature),
                length_penalty=float(length_penalty),
                uncertainty_penalty=0.0,
                uncertainty_column=cols["uncertainty"],
            )
            scores_hcr = build_edge_energies(graph, edges, cfg_hcr)
            scores_hgt = build_edge_energies(graph, edges, cfg_hgt)

            for _, q in queries.iterrows():
                true_path = parse_true_path(q["true_path"])
                hidden = parse_hidden_edge(q.get("hidden_edge"), true_path)
                mh = evaluate_query(
                    str(q["query_id"]),
                    str(q["source"]),
                    str(q["endpoint"]),
                    true_path,
                    hidden,
                    graph,
                    scores_hcr,
                    hubs,
                    top_k=top_k,
                )
                mg = evaluate_query(
                    str(q["query_id"]),
                    str(q["source"]),
                    str(q["endpoint"]),
                    true_path,
                    hidden,
                    graph,
                    scores_hgt,
                    hubs,
                    top_k=top_k,
                )
                mrr_hcr.append(float(mh["reciprocal_rank_at_100"]))
                mrr_hgt.append(float(mg["reciprocal_rank_at_100"]))
                tpm_hcr.append(float(mh["true_path_mass"]))
                hub_hcr.append(float(mh["hub_mass"]))
                hub_hgt.append(float(mg["hub_mass"]))
                if int(mh["true_path_hit_at_1"]) == 1:
                    ent_top1.append(float(mh["path_entropy"]))
                n_eval += 1

        mean_hub_hcr = float(sum(hub_hcr) / len(hub_hcr)) if hub_hcr else 0.0
        mean_hub_hgt = float(sum(hub_hgt) / len(hub_hgt)) if hub_hgt else 0.0
        rejected = mean_hub_hcr > mean_hub_hgt + 0.05
        grid_rows.append(
            {
                "temperature": float(temperature),
                "length_penalty": float(length_penalty),
                "n_eval": n_eval,
                "mrr_hcr": float(sum(mrr_hcr) / len(mrr_hcr)) if mrr_hcr else 0.0,
                "mrr_hgt": float(sum(mrr_hgt) / len(mrr_hgt)) if mrr_hgt else 0.0,
                "tpm_hcr": float(sum(tpm_hcr) / len(tpm_hcr)) if tpm_hcr else 0.0,
                "hub_hcr": mean_hub_hcr,
                "hub_hgt": mean_hub_hgt,
                "entropy_when_hit1": float(sum(ent_top1) / len(ent_top1))
                if ent_top1
                else float("nan"),
                "rejected_hub_mass": rejected,
            }
        )
        print(
            f"grid T={temperature} λL={length_penalty} "
            f"MRR={grid_rows[-1]['mrr_hcr']:.4f} rejected={rejected}",
            flush=True,
        )

    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(out_dir / "validation_grid.csv", index=False)

    candidates = grid_df[~grid_df["rejected_hub_mass"]].copy()
    if candidates.empty:
        candidates = grid_df.copy()

    # Maximize MRR@100; tie-break TPM; then lower entropy on top-1 hits.
    candidates = candidates.sort_values(
        by=["mrr_hcr", "tpm_hcr", "entropy_when_hit1"],
        ascending=[False, False, True],
    )
    # Explicit 0.002 MRR tie window → prefer higher TPM
    best = candidates.iloc[0]
    top_mrr = float(best["mrr_hcr"])
    near = candidates[candidates["mrr_hcr"] >= top_mrr - 0.002]
    if len(near) > 1:
        near = near.sort_values(
            by=["tpm_hcr", "entropy_when_hit1"],
            ascending=[False, True],
        )
        best = near.iloc[0]

    selected = {
        "temperature": float(best["temperature"]),
        "length_penalty": float(best["length_penalty"]),
        "uncertainty_penalty": 0.0,
        "tune_split": tune_split,
        "selection_rule": "max MRR@100; TPM tiebreak within 0.002; then lower entropy@hit1",
        "validation_mrr_hcr": float(best["mrr_hcr"]),
        "validation_tpm_hcr": float(best["tpm_hcr"]),
        "validation_hub_hcr": float(best["hub_hcr"]),
        "validation_hub_hgt": float(best["hub_hgt"]),
    }
    (out_dir / "selected_energy_config.json").write_text(
        json.dumps(selected, indent=2)
    )
    print(json.dumps(selected, indent=2))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
