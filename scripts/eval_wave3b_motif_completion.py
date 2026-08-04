#!/usr/bin/env python3
"""Evaluate overall + held-out dual-gate motif AUPRC (Wave 3B A1).

Motif edges are held out of G_train message passing but remain in the frozen
candidate split (train/valid/test). Primary diagnostic metric pools the 10
held-out parent_a→gate positives across splits and scores them against
validation negatives (fixed prevalence reference).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from sklearn.metrics import average_precision_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from data.PreprocessingTaskA.load_hetero_recon_data import load_recon_heterodata
from experiments.motif_completion import oracle_z_map
from hcr.attach_wave3b import fit_and_attach_hcr_wave3b
from models.TaskA.hetero_gnn import HeteroReconGNN


def compose_cfg(hcr: str, seed: int, motif: bool = True):
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
        f"experiment.motif_completion.enabled={'true' if motif else 'false'}",
        "experiment.motif_completion.hide=parent_a",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name="config", overrides=overrides)


@torch.no_grad()
def score(model, data, device):
    model.eval()
    data = data.to(device)
    logits = model(data).view(-1)
    probs = torch.sigmoid(logits).cpu().numpy()
    labels = data.edge_label.cpu().numpy().astype(float).reshape(-1)
    sources = list(data.candidate_source_name)
    targets = list(data.candidate_target_name)
    return labels, probs, sources, targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hcr", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = compose_cfg(args.hcr, args.seed, motif=True)
    train, valid, test, _ = load_recon_heterodata(cfg)
    enc = fit_and_attach_hcr_wave3b(cfg, train, valid, test, device="cpu")
    if enc is not None and hasattr(cfg.model, "decoder") and cfg.model.decoder is not None:
        OmegaConf.set_struct(cfg, False)
        cfg.model.decoder.hcr_dim = int(
            getattr(train, "hcr_dim", getattr(enc.config, "output_dim", 8))
        )
        OmegaConf.set_struct(cfg, True)

    device = torch.device("cpu")
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = HeteroReconGNN(
        cfg,
        train,
        in_channels=int(train[train.node_types[0]].x.size(-1)),
        hidden_channels=int(getattr(cfg.model, "hidden_channels", 64)),
    ).to(device)
    with torch.no_grad():
        _ = model(train.to(device))
    model.load_state_dict(payload["model_state_dict"])

    motif_pairs = set(oracle_z_map(cfg).keys())

    rows = []
    motif_probs = []
    motif_labels = []
    motif_meta = []

    for split, data in (("train", train), ("valid", valid), ("test", test)):
        labels, probs, sources, targets = score(model, data, device)
        a_all = float(average_precision_score(labels, probs))
        brier = float(brier_score_loss(labels, probs))
        for i, (s, t) in enumerate(zip(sources, targets)):
            if (s, t) in motif_pairs:
                motif_probs.append(float(probs[i]))
                motif_labels.append(float(labels[i]))
                motif_meta.append({"split": split, "source": s, "target": t, "prob": float(probs[i]), "label": float(labels[i])})
        rows.append(
            {
                "split": split,
                "hcr": args.hcr,
                "seed": args.seed,
                "auprc_all": a_all,
                "brier_all": brier,
                "n": int(len(labels)),
                "n_pos": int(labels.sum()),
            }
        )

    # Pool: all held-out motif positives + validation negatives
    _, valid_probs, _, _ = score(model, valid, device)
    valid_labels = valid.edge_label.cpu().numpy().astype(float).reshape(-1)
    valid_neg_probs = valid_probs[valid_labels < 0.5]
    pool_y = np.concatenate([np.ones(len(motif_labels)), np.zeros(len(valid_neg_probs))])
    pool_p = np.concatenate([np.asarray(motif_probs, dtype=float), valid_neg_probs])
    if len(motif_labels) and motif_labels and sum(motif_labels) > 0:
        auprc_motif = float(average_precision_score(pool_y, pool_p))
    else:
        auprc_motif = float("nan")

    summary = {
        "hcr": args.hcr,
        "seed": args.seed,
        "n_motif_edges_found": len(motif_labels),
        "n_motif_positives": int(sum(motif_labels)),
        "auprc_motif_heldout_vs_valid_neg": auprc_motif,
        "mean_prob_motif": float(np.mean(motif_probs)) if motif_probs else float("nan"),
        "splits": rows,
        "motif_edges": motif_meta,
        "checkpoint": args.checkpoint,
    }
    out = Path(args.out) if args.out else Path(args.checkpoint).with_suffix(".motif_metrics.json")
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in summary if k != "motif_edges"}, indent=2))


if __name__ == "__main__":
    main()
