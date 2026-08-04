#!/usr/bin/env python3
"""Evaluate held-out A→G motif AUPRC for Wave 4C latent-gate runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from sklearn.metrics import average_precision_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.PreprocessingTaskA.load_hetero_recon_data import load_recon_heterodata
from experiments.motif_completion import latent_triple_map
from hcr.attach_wave4c import fit_and_attach_hcr_wave4c
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
        "experiment.wave=WAVE4C_LATENT",
        "experiment.motif_completion.enabled=true",
        "experiment.motif_completion.hide=parent_a",
        "experiment.latent_gate.enabled=true",
    ]
    if hcr == "hcr3_random_context":
        overrides.append("hcr.z_mode=random_matched")
    elif hcr.startswith("hcr3_") or hcr == "latent_pairwise_aby":
        overrides.append("hcr.z_mode=oracle_outcome")
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name="config", overrides=overrides)


@torch.no_grad()
def score(model, data, device):
    model.eval()
    data = data.to(device)
    probs = torch.sigmoid(model(data).view(-1)).cpu().numpy()
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

    cfg = compose_cfg(args.hcr, args.seed)
    train, valid, test, _ = load_recon_heterodata(cfg)
    enc = fit_and_attach_hcr_wave4c(cfg, train, valid, test, device="cpu")
    if enc is not None and hasattr(cfg.model, "decoder") and cfg.model.decoder is not None:
        OmegaConf.set_struct(cfg, False)
        cfg.model.decoder.hcr_dim = int(
            getattr(train, "hcr_dim", getattr(getattr(enc, "config", None), "output_dim", 8))
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

    motif_pairs = set(latent_triple_map(cfg).keys())

    motif_probs, motif_labels, motif_meta, rows = [], [], [], []
    for split, data in (("train", train), ("valid", valid), ("test", test)):
        labels, probs, sources, targets = score(model, data, device)
        for i, (s, t) in enumerate(zip(sources, targets)):
            if (s, t) in motif_pairs:
                motif_probs.append(float(probs[i]))
                motif_labels.append(float(labels[i]))
                motif_meta.append(
                    {
                        "split": split,
                        "source": s,
                        "target": t,
                        "prob": float(probs[i]),
                        "label": float(labels[i]),
                    }
                )
        rows.append(
            {
                "split": split,
                "hcr": args.hcr,
                "seed": args.seed,
                "auprc_all": float(average_precision_score(labels, probs)),
                "brier_all": float(brier_score_loss(labels, probs)),
                "n": int(len(labels)),
                "n_pos": int(labels.sum()),
            }
        )

    _, valid_probs, _, _ = score(model, valid, device)
    valid_labels = valid.edge_label.cpu().numpy().astype(float).reshape(-1)
    valid_neg = valid_probs[valid_labels < 0.5]
    pool_y = np.concatenate([np.ones(len(motif_labels)), np.zeros(len(valid_neg))])
    pool_p = np.concatenate([np.asarray(motif_probs, dtype=float), valid_neg])
    auprc_motif = (
        float(average_precision_score(pool_y, pool_p)) if motif_labels else float("nan")
    )

    summary = {
        "wave": "wave4c_latent",
        "hcr": args.hcr,
        "seed": args.seed,
        "n_motif_edges_found": len(motif_labels),
        "n_motif_positives": int(sum(motif_labels)),
        "auprc_motif_heldout_vs_valid_neg": auprc_motif,
        "mean_prob_motif": float(np.mean(motif_probs)) if motif_probs else float("nan"),
        "splits": rows,
        "motif_edges": motif_meta,
        "latent_triples": {
            f"{a}->{g}": list(t) for (a, g), t in latent_triple_map(cfg).items()
        },
        "checkpoint": args.checkpoint,
    }
    out = Path(args.out) if args.out else Path(args.checkpoint).with_suffix(".latent_metrics.json")
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in summary if k not in {"motif_edges", "latent_triples"}}, indent=2))


if __name__ == "__main__":
    main()
