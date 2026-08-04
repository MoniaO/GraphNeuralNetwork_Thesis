#!/usr/bin/env python3
"""Wave 5B Panel E — energy variants on fixed union graph G_fixed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcr.motifs import load_truth_graph
from wnerw.wave5b.energy_maps import (
    EnergyConfig,
    add_matched_edgewise_shuffle,
    build_edge_energies,
    length_only_edge_scores,
    uniform_edge_scores,
)
from wnerw.wave5b.graph_builders import (
    SelectionSpec,
    build_fixed_union_graph,
    hub_nodes_from_degree,
)
from wnerw.wave5b.io import load_edges, load_queries, merge_patient_shuffle, resolve_path
from wnerw.wave5b.paired_summary import summarize_panel_e
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
    parser.add_argument(
        "--selected-config",
        type=Path,
        default=None,
        help="JSON from tune_wave5b_energy.py",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Override evaluation split (default: final_split from config)",
    )
    args = parser.parse_args()
    cfg = load_cfg(args.config)

    selected_path = args.selected_config or (
        ROOT / cfg["paths"]["output_dir"] / "tuning" / "selected_energy_config.json"
    )
    selected = json.loads(selected_path.read_text())
    temperature = float(selected["temperature"])
    length_penalty = float(selected["length_penalty"])
    # Uncertainty ablation uses first nonzero from grid if present, else 0.25.
    unc_grid = list(cfg["energy"].get("uncertainty_penalties", [0.0, 0.25]))
    unc_pen = next((float(x) for x in unc_grid if float(x) > 0), 0.25)

    out_dir = ROOT / cfg["paths"]["output_dir"] / "panel_e"
    out_dir.mkdir(parents=True, exist_ok=True)

    split = args.split or str(cfg["evaluation"]["final_split"])
    queries_all = load_queries(ROOT / cfg["paths"]["query_registry"])
    queries = queries_all[queries_all["query_split"] == split].copy()
    nodes, _ = load_truth_graph(None)
    thr = cfg["graph_selection"]
    cols = cfg["columns"]
    top_k = int(cfg["evaluation"]["top_k"])
    q_hub = float(cfg["hub_definition"]["degree_quantile"])

    rows = []
    for seed in cfg["seeds"]:
        edge_path = resolve_path(cfg["paths"]["edge_pattern"], seed, ROOT)
        ps_path = resolve_path(cfg["paths"]["patient_shuffle_pattern"], seed, ROOT)
        edges = merge_patient_shuffle(
            load_edges(edge_path),
            ps_path,
            out_col=cols["patient_shuffle_probability"],
        )
        edges = add_matched_edgewise_shuffle(
            edges,
            source_column=cols["hcr_probability"],
            output_column="p_hcr2_edgewise_shuffled",
            seed=int(seed),
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

        energy_variants: dict[str, dict | None] = {
            "E0_uniform": None,
            "E1_hgt": EnergyConfig(
                probability_column=cols["hgt_probability"],
                temperature=temperature,
                length_penalty=length_penalty,
            ),
            "E2_hcr2": EnergyConfig(
                probability_column=cols["hcr_probability"],
                temperature=temperature,
                length_penalty=length_penalty,
            ),
            "E3_hcr2_uncertainty": EnergyConfig(
                probability_column=cols["hcr_probability"],
                temperature=temperature,
                length_penalty=length_penalty,
                uncertainty_penalty=unc_pen,
                uncertainty_column=cols["uncertainty"],
            ),
            "E4_patient_shuffle": EnergyConfig(
                probability_column=cols["patient_shuffle_probability"],
                temperature=temperature,
                length_penalty=length_penalty,
            ),
            "E5_edgewise_shuffle": EnergyConfig(
                probability_column="p_hcr2_edgewise_shuffled",
                temperature=temperature,
                length_penalty=length_penalty,
            ),
        }

        score_maps = {}
        for name, ecfg in energy_variants.items():
            if ecfg is None:
                if length_penalty > 0:
                    score_maps[name] = length_only_edge_scores(
                        graph,
                        length_penalty=length_penalty,
                        temperature=temperature,
                    )
                else:
                    score_maps[name] = uniform_edge_scores(graph)
            else:
                score_maps[name] = build_edge_energies(graph, edges, ecfg)

        for variant, scores in score_maps.items():
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
                )
                rows.append(
                    {
                        "variant": variant,
                        "seed": int(seed),
                        "query_split": split,
                        "temperature": temperature,
                        "length_penalty": length_penalty,
                        **metrics,
                    }
                )
        print(f"Panel E seed={seed} split={split} done", flush=True)

    df = pd.DataFrame(rows)
    out_csv = out_dir / f"query_metrics_{split}.csv"
    df.to_csv(out_csv, index=False)
    summary = summarize_panel_e(df)
    summary.to_csv(out_dir / f"summary_{split}.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
