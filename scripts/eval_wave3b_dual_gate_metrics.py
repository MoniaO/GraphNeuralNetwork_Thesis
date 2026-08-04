#!/usr/bin/env python3
"""Wave 3B dual-gate metrics: AUPRC, MeanRank, Recall@K, Δshuffle helper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from data.PreprocessingTaskA.load_hetero_recon_data import load_recon_heterodata
from experiments.motif_completion import oracle_z_map
from hcr.attach_wave3b import fit_and_attach_hcr_wave3b
from models.TaskA.hetero_gnn import HeteroReconGNN


def compose_cfg(hcr: str, seed: int):
    overrides = [
        "model=TaskA_hgt_hcr" if hcr != "none" else "model=TaskA_hgt",
        f"hcr={hcr}" if hcr != "none" else "hcr=none",
        "model.num_layers=1",
        "model.heads=8",
        "model.hidden_dim=64",
        "model.hidden_channels=64",
        f"training.seed={seed}",
        "training.device=cpu",
        "data.dataset.scenario=clean",
        "data.feature_ablation_profile=empirical",
        "data.candidate_seed=20260722",
        "wandb.enabled=false",
        "experiment.wave=WAVE3B_MOTIF",
        "experiment.motif_completion.enabled=true",
        "experiment.motif_completion.hide=parent_a",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name="config", overrides=overrides)


@torch.no_grad()
def collect(model, data, device, motif_pairs):
    model.eval()
    data = data.to(device)
    probs = torch.sigmoid(model(data).view(-1)).cpu().numpy()
    labels = data.edge_label.cpu().numpy().astype(float).reshape(-1)
    sources = list(data.candidate_source_name)
    targets = list(data.candidate_target_name)
    rows = []
    for i, (s, t) in enumerate(zip(sources, targets)):
        rows.append(
            {
                "source": s,
                "target": t,
                "prob": float(probs[i]),
                "label": float(labels[i]),
                "is_motif": (s, t) in motif_pairs,
            }
        )
    return rows


def mean_rank_and_recall(rows, ks=(5, 10, 20)):
    """Rank each motif positive among all candidates (pooled), report mean rank + recall@K."""
    # Global ranking by probability descending
    order = sorted(rows, key=lambda r: r["prob"], reverse=True)
    rank_of = {}
    for rank, r in enumerate(order, start=1):
        key = (r["source"], r["target"])
        if key not in rank_of:
            rank_of[key] = rank
    motifs = [r for r in rows if r["is_motif"] and r["label"] > 0.5]
    if not motifs:
        return {"mean_rank": float("nan"), "n_motif": 0, **{f"recall_at_{k}": float("nan") for k in ks}}
    ranks = [rank_of[(m["source"], m["target"])] for m in motifs]
    out = {
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "n_motif": len(motifs),
        "n_candidates": len(rows),
    }
    for k in ks:
        out[f"recall_at_{k}"] = float(np.mean([1.0 if r <= k else 0.0 for r in ranks]))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hcr", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = compose_cfg(args.hcr, args.seed)
    train, valid, test, _ = load_recon_heterodata(cfg)
    enc = fit_and_attach_hcr_wave3b(cfg, train, valid, test, device="cpu")
    if enc is not None and hasattr(cfg.model, "decoder") and cfg.model.decoder is not None:
        OmegaConf.set_struct(cfg, False)
        cfg.model.decoder.hcr_dim = int(getattr(train, "hcr_dim", 8))
        OmegaConf.set_struct(cfg, True)

    device = torch.device("cpu")
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = HeteroReconGNN(
        cfg,
        train,
        in_channels=int(train[train.node_types[0]].x.size(-1)),
        hidden_channels=64,
    ).to(device)
    with torch.no_grad():
        _ = model(train.to(device))
    model.load_state_dict(payload["model_state_dict"])

    motif_pairs = set(oracle_z_map(cfg).keys())
    all_rows = []
    split_metrics = {}
    for name, data in (("train", train), ("valid", valid), ("test", test)):
        rows = collect(model, data, device, motif_pairs)
        all_rows.extend([{**r, "split": name} for r in rows])
        y = np.array([r["label"] for r in rows])
        p = np.array([r["prob"] for r in rows])
        split_metrics[name] = {
            "auprc_all": float(average_precision_score(y, p)),
            "n": len(rows),
            "n_pos": int(y.sum()),
        }

    # Motif held-out AUPRC vs valid negatives
    motif = [r for r in all_rows if r["is_motif"] and r["label"] > 0.5]
    valid_neg = [r for r in all_rows if r["split"] == "valid" and r["label"] < 0.5]
    pool_y = np.array([1.0] * len(motif) + [0.0] * len(valid_neg))
    pool_p = np.array([r["prob"] for r in motif] + [r["prob"] for r in valid_neg])
    auprc_dual = (
        float(average_precision_score(pool_y, pool_p)) if len(motif) else float("nan")
    )

    rank_stats = mean_rank_and_recall(all_rows)
    summary = {
        "protocol": "observed_gate_completion",
        "hcr": args.hcr,
        "seed": args.seed,
        "auprc_all_valid": split_metrics["valid"]["auprc_all"],
        "auprc_all_test": split_metrics["test"]["auprc_all"],
        "auprc_dual_gate": auprc_dual,
        "n_motif_found": len(motif),
        **rank_stats,
        "splits": split_metrics,
        "checkpoint": args.checkpoint,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
