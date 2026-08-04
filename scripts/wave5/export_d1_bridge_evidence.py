#!/usr/bin/env python3
"""Wave 5B — export D1 structural-latent decoder evidence for the path bridge.

For each frozen Wave-4D D1 checkpoint (parent_a mask):

  P_D1(A→G)   from HGT + structural_latent_pairwise (24-d)
  P_HGT(A→G)  from frozen HGT baseline (same candidate split)
  Δlogit      = logit(P_D1) − logit(P_HGT)

Motif completion (hide=parent_a) is ON so held-out A→G edges are predicted
edges in G*, not near-certain train edges.

Writes:
  outputs/wave5/evidence_d1/seed{SEED}_d1_bridge.csv
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.PreprocessingTaskA.load_hetero_recon_data import load_recon_heterodata
from hcr.attach_wave4d import fit_and_attach_hcr_wave4d
from hcr.motifs import load_truth_graph
from models.TaskA.hetero_gnn import HeteroReconGNN
from wnerw.calibration import binary_entropy_uncertainty, fit_platt
from wnerw.graph_builder import annotate_topological_allowed, layer_rank_from_nodes
from wnerw.potentials import safe_logit

# Wave 4D D1 checkpoints (hide=parent_a) — structural_latent_pairwise.
D1_CKPT = {
    20260721: ROOT / "outputs/2026-07-31/21-12-49/best_model.pt",
    20260722: ROOT / "outputs/2026-07-31/21-18-16/best_model.pt",
    20260723: ROOT / "outputs/2026-07-31/21-23-15/best_model.pt",
}

# Frozen Wave-2 FINAL HGT L1 H8 (same as Wave 5A).
HGT_CKPT = {
    20260721: ROOT / "outputs/2026-07-31/14-56-55/best_model.pt",
    20260722: ROOT / "outputs/2026-07-31/14-58-21/best_model.pt",
    20260723: ROOT / "outputs/2026-07-31/15-00-08/best_model.pt",
}


def compose_cfg(*, d1: bool, seed: int):
    overrides = [
        "model=TaskA_hgt_hcr" if d1 else "model=TaskA_hgt",
        "hcr=structural_latent_pairwise" if d1 else "hcr=none",
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
        # wave4d_enabled() matches WAVE4D / CONTEXT_ROLE or causal_role_audit.
        "experiment.wave=WAVE5B_D1_BRIDGE",
        "experiment.motif_completion.enabled=true",
        "experiment.motif_completion.hide=parent_a",
        "experiment.causal_role_audit.enabled=true",
        "experiment.context_selection.mode=structural",
        "experiment.context_selection.source_graph=train",
        "experiment.context_selection.use_true_graph=false",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name="config", overrides=overrides)


@torch.no_grad()
def predict_split(model, data, device) -> pd.DataFrame:
    model.eval()
    data = data.to(device)
    logits = model(data).view(-1).cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    labels = data.edge_label.cpu().numpy().astype(np.float64).reshape(-1)
    sources = list(data.candidate_source_name)
    targets = list(data.candidate_target_name)
    return pd.DataFrame(
        {
            "candidate_id": [f"{s}->{t}" for s, t in zip(sources, targets)],
            "source": sources,
            "target": targets,
            "label": labels,
            "logit": logits.astype(np.float64),
            "p_raw": probs.astype(np.float64),
        }
    )


def load_model(cfg, train_data, ckpt: Path, device: torch.device):
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    model = HeteroReconGNN(
        cfg,
        train_data,
        in_channels=int(train_data[train_data.node_types[0]].x.size(-1)),
        hidden_channels=int(getattr(cfg.model, "hidden_channels", 64)),
    ).to(device)
    with torch.no_grad():
        _ = model(train_data.to(device))
    model.load_state_dict(payload["model_state_dict"])
    return model


def message_passing_pairs(train_data) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    name_maps = {}
    for ntype in train_data.node_types:
        store = train_data[ntype]
        if hasattr(store, "node_name"):
            name_maps[ntype] = [str(x) for x in store.node_name]
    for etype in train_data.edge_types:
        src_type, rel, dst_type = etype
        if str(rel).startswith("rev_"):
            continue
        edge_index = train_data[etype].edge_index
        if edge_index is None or edge_index.numel() == 0:
            continue
        src_names = name_maps.get(src_type)
        dst_names = name_maps.get(dst_type)
        if src_names is None or dst_names is None:
            continue
        for i in range(edge_index.size(1)):
            pairs.add((src_names[int(edge_index[0, i])], dst_names[int(edge_index[1, i])]))
    return pairs


def calibrate(df: pd.DataFrame, raw_col: str, cal_col: str) -> pd.DataFrame:
    valid = df[df["edge_split"] == "valid"]
    cal = fit_platt(valid[raw_col].to_numpy(), valid["label"].to_numpy())
    out = df.copy()
    out[cal_col] = cal.transform(out[raw_col].to_numpy())
    return out


def export_seed(seed: int, out_dir: Path, device: torch.device) -> Path:
    d1_ckpt = D1_CKPT[seed]
    hgt_ckpt = HGT_CKPT[seed]
    if not d1_ckpt.exists():
        raise FileNotFoundError(d1_ckpt)
    if not hgt_ckpt.exists():
        raise FileNotFoundError(hgt_ckpt)

    # Shared motif-completion split (hide=parent_a) for both models.
    print(f"\n=== seed {seed}: load motif-completion data ===")
    cfg_d1 = compose_cfg(d1=True, seed=seed)
    train, valid, test, _ = load_recon_heterodata(cfg_d1)
    enc = fit_and_attach_hcr_wave4d(cfg_d1, train, valid, test, device=device)
    if enc is not None and hasattr(cfg_d1.model, "decoder") and cfg_d1.model.decoder is not None:
        OmegaConf.set_struct(cfg_d1, False)
        cfg_d1.model.decoder.hcr_dim = int(getattr(train, "hcr_dim", 24))
        OmegaConf.set_struct(cfg_d1, True)

    print(f"=== seed {seed}: D1 structural_latent_pairwise ===")
    model_d1 = load_model(cfg_d1, train, d1_ckpt, device)
    d1_frames = []
    for split, data in (("train", train), ("valid", valid), ("test", test)):
        fr = predict_split(model_d1, data, device)
        fr["edge_split"] = split
        d1_frames.append(fr.rename(columns={"p_raw": "p_d1", "logit": "logit_d1"}))
    d1_df = pd.concat(d1_frames, ignore_index=True)

    # HGT on the same candidate HeteroData (strip HCR features).
    print(f"=== seed {seed}: HGT baseline on same candidates ===")
    cfg_hgt = compose_cfg(d1=False, seed=seed)
    train_h, valid_h, test_h, _ = load_recon_heterodata(cfg_hgt)
    for d in (train_h, valid_h, test_h):
        d.hcr_enabled = False
        d.hcr_features = None
    model_h = load_model(cfg_hgt, train_h, hgt_ckpt, device)
    hgt_frames = []
    for split, data in (("train", train_h), ("valid", valid_h), ("test", test_h)):
        fr = predict_split(model_h, data, device)
        fr["edge_split"] = split
        hgt_frames.append(fr.rename(columns={"p_raw": "p_hgt", "logit": "logit_hgt"}))
    hgt_df = pd.concat(hgt_frames, ignore_index=True)

    df = d1_df.merge(
        hgt_df[["candidate_id", "p_hgt", "logit_hgt"]],
        on="candidate_id",
        how="inner",
        validate="one_to_one",
    )
    if len(df) == 0:
        raise RuntimeError(f"No overlapping candidates for seed={seed}")

    df = calibrate(df, "p_hgt", "p_hgt_calibrated")
    df = calibrate(df, "p_d1", "p_d1_calibrated")
    df["p_calibrated"] = df["p_d1_calibrated"]
    df["delta_logit"] = [
        safe_logit(float(a)) - safe_logit(float(b))
        for a, b in zip(df["p_d1_calibrated"], df["p_hgt_calibrated"])
    ]
    df["q_d1"] = [
        math.log(max(min(float(p), 1.0 - 1e-6), 1e-6)) for p in df["p_d1_calibrated"]
    ]
    df["hcr_uncertainty"] = [
        binary_entropy_uncertainty(p) for p in df["p_d1_calibrated"].tolist()
    ]

    mp = message_passing_pairs(train)
    df["edge_in_train"] = [
        (str(s), str(t)) in mp for s, t in zip(df["source"], df["target"])
    ]

    nodes, _ = load_truth_graph(cfg_d1)
    ranks = layer_rank_from_nodes(nodes)
    df["topological_allowed"] = annotate_topological_allowed(df, ranks).to_numpy()

    ctx = getattr(train, "wave4d_contexts", {}) or {}
    df["structural_context"] = [
        ctx.get(f"{s}->{t}") for s, t in zip(df["source"], df["target"])
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"seed{seed}_d1_bridge.csv"
    df.to_csv(out_path, index=False)
    meta = {
        "seed": seed,
        "hide": "parent_a",
        "hcr_variant": "structural_latent_pairwise",
        "n_rows": int(len(df)),
        "n_overlap": int(len(df)),
        "d1_checkpoint": str(d1_ckpt),
        "hgt_checkpoint": str(hgt_ckpt),
        "n_train_mp_edges": int(len(mp)),
        "n_structural_contexts": int(sum(1 for v in ctx.values() if v is not None)),
        "calibration": "platt_on_valid",
        "table_path": str(out_path),
        "bridge": "q_AG = log P_D1(A→G); delta_logit = logit(P_D1)-logit(P_HGT)",
        "graph_note": "edge_in_train from motif-hidden G_train (parent_a held out)",
    }
    (out_dir / f"seed{seed}_d1_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out_path} ({len(df)} rows, contexts={meta['n_structural_contexts']})")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/wave5/evidence_d1",
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=list(D1_CKPT))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    for seed in args.seeds:
        export_seed(int(seed), args.out_dir, device)


if __name__ == "__main__":
    main()
