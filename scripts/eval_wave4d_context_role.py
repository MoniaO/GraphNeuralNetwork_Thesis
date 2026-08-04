#!/usr/bin/env python3
"""Wave 4D evaluation — Panel A motif recovery + Panel B causal-role audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.PreprocessingTaskA.load_hetero_recon_data import load_recon_heterodata
from data.patient_matrix import load_patient_matrix_with_split, train_patient_df
from experiments.motif_completion import hide_tasks
from hcr.attach_wave4d import fit_and_attach_hcr_wave4d, resolve_context_map
from hcr.context_roles import (
    build_context_role_registry,
    classify_selected_z_role,
    registry_to_frame,
    truth_digraph,
)
from hcr.motif_registry_v3 import GATE_OUTCOMES
from models.TaskA.hetero_gnn import HeteroReconGNN

VARIANT_MODE = {
    "none": "none",
    "structural_latent_pairwise": "structural",
    "all_context_top1": "all_context_top1",
    "structural_context_shuffled": "structural",
    "matched_random_context": "matched_random",
    "hcr3_selected_capacity_matched": "structural",
}


def compose_cfg(hcr: str, seed: int, hide: str):
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
        "experiment.wave=WAVE4D_CONTEXT_ROLE_AUDIT",
        "experiment.motif_completion.enabled=true",
        f"experiment.motif_completion.hide={hide}",
        "experiment.causal_role_audit.enabled=true",
        f"experiment.context_selection.mode={VARIANT_MODE.get(hcr, 'structural')}",
        "experiment.context_selection.source_graph=train",
        "experiment.context_selection.use_true_graph=false",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name="config", overrides=overrides)


@torch.no_grad()
def score_split(model, data, device):
    model.eval()
    data = data.to(device)
    probs = torch.sigmoid(model(data).view(-1)).cpu().numpy()
    labels = data.edge_label.cpu().numpy().astype(float).reshape(-1)
    sources = list(data.candidate_source_name)
    targets = list(data.candidate_target_name)
    return labels, probs, sources, targets


def ranking_metrics(motif_probs, valid_neg_probs, ks=(1, 3, 5)):
    """Rank each motif positive against the valid-negative pool."""
    if not motif_probs:
        return {
            "mrr": float("nan"),
            "mean_rank": float("nan"),
            "median_rank": float("nan"),
            **{f"hits_at_{k}": float("nan") for k in ks},
        }
    neg = np.asarray(valid_neg_probs, dtype=float)
    ranks = []
    for p in motif_probs:
        # rank = 1 + #neg with score > p  (ties broken pessimistically)
        ranks.append(1 + int(np.sum(neg > p)))
    ranks = np.asarray(ranks, dtype=float)
    out = {
        "mrr": float(np.mean(1.0 / ranks)),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "n_ranked": int(len(ranks)),
    }
    for k in ks:
        out[f"hits_at_{k}"] = float(np.mean(ranks <= k))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hcr", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--hide", required=True, choices=["parent_a", "parent_b"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = compose_cfg(args.hcr, args.seed, args.hide)
    train, valid, test, _ = load_recon_heterodata(cfg)
    enc = fit_and_attach_hcr_wave4d(cfg, train, valid, test, device="cpu")
    if enc is not None and hasattr(cfg.model, "decoder") and cfg.model.decoder is not None:
        OmegaConf.set_struct(cfg, False)
        cfg.model.decoder.hcr_dim = int(
            getattr(train, "hcr_dim", getattr(getattr(enc, "config", None), "output_dim", 24))
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

    patient_df = load_patient_matrix_with_split(cfg)
    train_patients = train_patient_df(patient_df)
    tasks = hide_tasks(cfg, train_patients)
    motif_pairs = {(t.candidate_source, t.candidate_target) for t in tasks}
    true_coparent = {(t.candidate_source, t.candidate_target): t.visible_parent for t in tasks}

    contexts = resolve_context_map(cfg, train, train_patients) if args.hcr != "none" else {}
    g_true = truth_digraph(cfg)

    # --- Panel A: motif recovery ---
    motif_probs, motif_labels, motif_meta = [], [], []
    valid_neg_probs = []
    for split, data in (("train", train), ("valid", valid), ("test", test)):
        labels, probs, sources, targets = score_split(model, data, device)
        for i, (s, t) in enumerate(zip(sources, targets)):
            if (s, t) in motif_pairs:
                motif_probs.append(float(probs[i]))
                motif_labels.append(float(labels[i]))
                z = contexts.get((s, t))
                y = GATE_OUTCOMES.get(t)
                motif_meta.append(
                    {
                        "split": split,
                        "source": s,
                        "target": t,
                        "prob": float(probs[i]),
                        "label": float(labels[i]),
                        "selected_z": z,
                        "true_coparent": true_coparent.get((s, t)),
                        "selected_z_role": classify_selected_z_role(
                            g_true, s, t, z, y
                        ),
                    }
                )
            elif split == "valid" and labels[i] < 0.5:
                valid_neg_probs.append(float(probs[i]))

    pool_y = np.concatenate([np.ones(len(motif_labels)), np.zeros(len(valid_neg_probs))])
    pool_p = np.concatenate([np.asarray(motif_probs, dtype=float), np.asarray(valid_neg_probs)])
    auprc_motif = (
        float(average_precision_score(pool_y, pool_p)) if motif_labels else float("nan")
    )
    panel_a = {
        "auprc_motif": auprc_motif,
        **ranking_metrics(motif_probs, valid_neg_probs),
        "n_motif_edges": len(motif_labels),
        "n_motif_positives": int(sum(motif_labels)),
        "mean_prob_motif": float(np.mean(motif_probs)) if motif_probs else float("nan"),
    }

    # ContextSelectionPrecision@1
    if contexts:
        hits = sum(
            1
            for key, z in contexts.items()
            if z is not None and z == true_coparent.get(key)
        )
        precision_at_1 = hits / max(len(contexts), 1)
    else:
        precision_at_1 = float("nan")

    selected_role_counts: dict[str, int] = {}
    for m in motif_meta:
        role = m.get("selected_z_role", "none")
        selected_role_counts[role] = selected_role_counts.get(role, 0) + 1

    # --- Panel B: role registry (G_true evaluator only) ---
    cra = cfg.experiment.causal_role_audit
    registry = build_context_role_registry(
        cfg,
        train_patients=train_patients,
        max_per_role=int(getattr(cra, "max_per_role", 50)),
        seed=args.seed,
        roles=list(getattr(cra, "roles", [])),
    )
    # Score registry pairs that appear in candidate pools.
    score_lookup: dict[tuple[str, str], float] = {}
    for data in (train, valid, test):
        labels, probs, sources, targets = score_split(model, data, device)
        for s, t, p in zip(sources, targets, probs):
            score_lookup[(s, t)] = float(p)

    by_role: dict[str, list[float]] = {}
    for rec in registry:
        p = score_lookup.get((rec.source, rec.target))
        if p is None:
            continue
        by_role.setdefault(rec.role, []).append(p)

    pos_scores = by_role.get("true_coparent", [])
    thr = 0.5
    panel_b: dict[str, dict] = {}
    for role, scores in by_role.items():
        arr = np.asarray(scores, dtype=float)
        entry = {
            "n_scored": int(len(arr)),
            "n_registry": int(sum(1 for r in registry if r.role == role)),
            "mean_score": float(arr.mean()) if len(arr) else float("nan"),
            "fpr_at_0.5": float(np.mean(arr >= thr)) if len(arr) and role != "true_coparent" else float("nan"),
            "descriptive_only": bool(len(arr) < 10),
        }
        if role != "true_coparent" and pos_scores and scores:
            y = np.array([1.0] * len(pos_scores) + [0.0] * len(scores))
            p = np.array(list(pos_scores) + list(scores))
            entry["auprc_true_coparent_vs_role"] = float(average_precision_score(y, p))
        panel_b[role] = entry

    summary = {
        "wave": "wave4d_context_role_audit",
        "hcr": args.hcr,
        "seed": args.seed,
        "hide": args.hide,
        "panel_a": panel_a,
        "panel_b": panel_b,
        "context_selection_precision_at_1": precision_at_1,
        "selected_z_role_counts": selected_role_counts,
        "contexts": {f"{a}->{g}": z for (a, g), z in contexts.items()},
        "motif_edges": motif_meta,
        "checkpoint": args.checkpoint,
    }
    out = Path(args.out) if args.out else Path(args.checkpoint).with_suffix(".wave4d_metrics.json")
    out.write_text(json.dumps(summary, indent=2))
    # Also dump registry once beside metrics.
    reg_path = out.with_name(f"role_registry_{args.hide}_{args.seed}.csv")
    if not reg_path.exists():
        registry_to_frame(registry).to_csv(reg_path, index=False)

    print(
        json.dumps(
            {
                "hcr": args.hcr,
                "seed": args.seed,
                "hide": args.hide,
                "auprc_motif": panel_a["auprc_motif"],
                "mrr": panel_a["mrr"],
                "hits_at_1": panel_a["hits_at_1"],
                "context_selection_precision_at_1": precision_at_1,
                "selected_z_role_counts": selected_role_counts,
                "panel_b_fpr": {
                    r: panel_b[r].get("fpr_at_0.5")
                    for r in panel_b
                    if r != "true_coparent"
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
