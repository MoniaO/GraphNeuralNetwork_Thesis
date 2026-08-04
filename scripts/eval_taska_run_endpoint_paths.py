#!/usr/bin/env python3
"""Endpoint-path metrics for one Task A MLP/KAN run (valid + test)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
WAVE_OUT = ROOT / "outputs" / "wave11_taskA"
OUT = WAVE_OUT / "mlp_vs_kan_full"
DEFAULT_GSN = Path.home() / "Desktop" / "GSN Graphs dysertation 2026"
CONFIG_DIR = str(ROOT / "configs")

MODEL_CFG = {
    "TA_MLP": ROOT / "configs" / "taskA" / "mlp_full_retrain.yaml",
    "TA_KAN": ROOT / "configs" / "taskA" / "kan_full_retrain.yaml",
    "TA_KAN_SHALLOW": (
        ROOT / "configs" / "taskA" / "kan_architectures" / "k1_shallow.yaml"
    ),
}


def _overrides(model_id: str, scenario: str, seed: int) -> list[str]:
    cfg = yaml.safe_load(MODEL_CFG[model_id].read_text())
    pe = cfg["decoder"]["pair_encoder"]
    pe_type = str(pe["type"])
    out = [
        f"model={cfg['model']}",
        f"hcr={cfg['hcr']}",
        f"data.dataset.scenario={scenario}",
        "data.feature_ablation_profile=empirical",
        f"data.candidate_seed={seed}",
        f"training.seed={seed}",
        "training.device=cpu",
        "experiment.wave=TASKA_MLP_VS_KAN",
        f"experiment.variant={cfg['variant']}",
        "model.decoder.arch=unshared_mlp",
        "model.decoder.ablation=none",
        "model.decoder.use_triple=false",
        "model.decoder.ag_kan_residual.enabled=false",
        f"model.decoder.pair_encoder.type={pe_type}",
        "model.decoder.pair_encoder.input_dim=40",
        "model.decoder.pair_encoder.hidden_dim=16",
        "model.decoder.pair_encoder.output_dim=8",
        f"model.decoder.pair_encoder.dropout={pe.get('dropout', 0.1)}",
        "model.hgt.hidden_dim=32",
        "model.hgt.num_layers=3",
        "model.hgt.heads=8",
        "model.hgt.activation=leaky_relu",
        "model.hgt.dropout=0.2",
        "model.hgt.residual=true",
        "model.num_layers=3",
        "model.hidden_dim=32",
        "model.hidden_channels=32",
        "model.heads=8",
        "wandb.enabled=false",
    ]
    if pe_type != "mlp":
        gr = pe.get("grid_range", [-3.0, 3.0])
        out += [
            f"model.decoder.pair_encoder.spline_order={pe.get('spline_order', 3)}",
            f"model.decoder.pair_encoder.grid_size={pe.get('grid_size', 5)}",
            f"model.decoder.pair_encoder.grid_range=[{gr[0]},{gr[1]}]",
            "model.decoder.pair_encoder.grid_update=false",
            f"model.decoder.pair_encoder.base_activation={pe.get('base_activation', 'silu')}",
            f"model.decoder.pair_encoder.base_scale_init={pe.get('base_scale_init', 1.0)}",
            f"model.decoder.pair_encoder.spline_scale_init={pe.get('spline_scale_init', 0.1)}",
            "model.decoder.pair_encoder.use_bias=true",
            f"model.decoder.pair_encoder.spline_l1={pe.get('spline_l1', 1.0e-5)}",
        ]
    return out


@torch.no_grad()
def _predict(model, data, device) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    model.eval()
    data = data.to(device)
    logits = model(data).view(-1)
    probs = torch.sigmoid(logits).cpu().numpy()
    labels = data.edge_label.float().cpu().numpy()
    src = list(data.candidate_source_name)
    tgt = list(data.candidate_target_name)
    return labels, probs, src, tgt


def main() -> None:
    from data.PreprocessingTaskA.load_hetero_recon_data import load_recon_heterodata
    from evaluation.taskA_endpoint_reachability import (
        DEFAULT_ENDPOINTS,
        build_node_endpoint_table,
        endpoint_subset_metrics,
        expand_candidates_to_endpoint_registry,
        load_audited_digraph,
    )
    from hcr.wave7.wave7c.attach import fit_and_attach_wave7c
    from train_taskA import build_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_CFG))
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    if args.model == "TA_KAN_SHALLOW":
        run_dir = (
            ROOT
            / "outputs"
            / "wave11_taskA"
            / "kan_architecture_audit"
            / "scenarios"
            / "runs"
            / args.model
            / args.scenario
            / f"seed_{args.seed}"
        )
    else:
        run_dir = OUT / "runs" / args.model / args.scenario / f"seed_{args.seed}"
    ckpt_path = run_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"missing {ckpt_path}")

    os.environ.setdefault("GSN_PROJECT_ROOT", str(DEFAULT_GSN))
    gsn = Path(os.environ["GSN_PROJECT_ROOT"])
    data_dir = gsn / "2 v3. Data" / "dataset_v3"

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=_overrides(args.model, args.scenario, args.seed))

    train_data, valid_data, test_data, _node_to_idx = load_recon_heterodata(cfg)
    fit_and_attach_wave7c(cfg, train_data, valid_data, test_data, device="cpu")
    model = build_model(cfg, train_data)
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model_state_dict"])
    device = torch.device("cpu")
    model.to(device)

    node_ep_path = OUT / "node_endpoint_reachability.csv"
    if node_ep_path.exists():
        node_ep = pd.read_csv(node_ep_path)
    else:
        graph = load_audited_digraph(
            data_dir / "synthetic_pharmacotherapy_v3_edges_audited.csv",
            nodes_csv=data_dir / "synthetic_pharmacotherapy_v3_nodes.csv",
        )
        node_ep = build_node_endpoint_table(graph, DEFAULT_ENDPOINTS)

    rows = []
    # Structural Task A candidates are shared; hospital-specific rows use the same
    # logits tagged for multihospital audit tables (pooled + per-hospital labels).
    hospitals = ["pooled"]
    if args.scenario == "multihospital":
        samples = pd.read_csv(
            data_dir / "synthetic_pharmacotherapy_v3_samples_multihospital.csv",
            usecols=["hospital_id"],
        )
        if "hospital_id" in samples.columns:
            hospitals = ["pooled"] + [
                f"hospital_{h}" for h in sorted(samples["hospital_id"].astype(str).unique())
            ]

    for split_name, data in (("valid", valid_data), ("test", test_data)):
        labels, probs, src, tgt = _predict(model, data, device)
        frame = pd.DataFrame(
            {
                "source": src,
                "target": tgt,
                "edge_label": labels,
                "probability": probs,
                "split": split_name,
            }
        )
        frame["candidate_id"] = [f"{s}__{t}" for s, t in zip(src, tgt)]
        reg = expand_candidates_to_endpoint_registry(frame, node_ep)
        pred = frame.merge(
            reg,
            left_on=["candidate_id"],
            right_on=["candidate_edge_id"],
            how="inner",
            suffixes=("", "_reg"),
        )

        for hospital in hospitals:
            for ep in DEFAULT_ENDPOINTS:
                sub = pred[
                    (pred["endpoint"] == ep)
                    & (pred["is_downstream_reachable"].astype(bool))
                ]
                metrics = endpoint_subset_metrics(
                    sub["edge_label"].to_numpy(),
                    sub["probability"].to_numpy(),
                    sub["shortest_path_length"].to_numpy(),
                )
                rows.append(
                    {
                        "model": args.model,
                        "scenario": args.scenario,
                        "seed": args.seed,
                        "hospital": hospital,
                        "endpoint": ep,
                        "split": split_name,
                        "score_source": "pooled_structural_logits",
                        **metrics,
                    }
                )

        pred_out = run_dir / f"predictions_{split_name}.csv"
        frame.to_csv(pred_out, index=False)

    out_csv = run_dir / "endpoint_path_metrics.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"wrote {out_csv} n={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
