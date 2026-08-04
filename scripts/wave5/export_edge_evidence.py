#!/usr/bin/env python3
"""Export calibrated edge evidence tables for Wave 5 WNERW (5 frozen seeds).

Writes:
  outputs/wave5/evidence/seed{SEED}_edges.parquet

Does not open G_true for scoring — only audited nodes (layer ranks), G_train
message-passing edges, and frozen candidate splits + model checkpoints.
"""

from __future__ import annotations

import argparse
import json
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
from hcr.attach import fit_and_attach_hcr
from models.TaskA.hetero_gnn import HeteroReconGNN
from wnerw.calibration import binary_entropy_uncertainty, fit_platt
from wnerw.graph_builder import annotate_topological_allowed, layer_rank_from_nodes

SEEDS = (20260721, 20260722, 20260723, 20260724, 20260725)

# Frozen Wave-2 FINAL HGT L1 H8 checkpoints (heads=8 runs).
HGT_CKPT = {
    20260721: ROOT / "outputs/2026-07-31/14-56-55/best_model.pt",
    20260722: ROOT / "outputs/2026-07-31/14-58-21/best_model.pt",
    20260723: ROOT / "outputs/2026-07-31/15-00-08/best_model.pt",
    20260724: ROOT / "outputs/2026-07-31/15-01-43/best_model.pt",
    20260725: ROOT / "outputs/2026-07-31/15-03-32/best_model.pt",
}

# Frozen Wave-3A Stage-2 binary_compact checkpoints.
HCR2_CKPT = {
    20260721: ROOT / "outputs/2026-07-31/15-48-14/best_model.pt",
    20260722: ROOT / "outputs/2026-07-31/15-52-24/best_model.pt",
    20260723: ROOT / "outputs/2026-07-31/15-57-05/best_model.pt",
    20260724: ROOT / "outputs/2026-07-31/16-16-19/best_model.pt",
    20260725: ROOT / "outputs/2026-07-31/16-18-42/best_model.pt",
}


def compose_cfg(*, hcr: str, seed: int):
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
        "experiment.wave=WAVE5_EVIDENCE",
        "experiment.motif_completion.enabled=false",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name="config", overrides=overrides)


@torch.no_grad()
def predict_split(model, data, device) -> pd.DataFrame:
    model.eval()
    data = data.to(device)
    probs = torch.sigmoid(model(data).view(-1)).cpu().numpy()
    labels = data.edge_label.cpu().numpy().astype(np.float64).reshape(-1)
    sources = list(data.candidate_source_name)
    targets = list(data.candidate_target_name)
    edge_types = (
        list(data.candidate_edge_type)
        if hasattr(data, "candidate_edge_type") and data.candidate_edge_type is not None
        else ["unknown"] * len(sources)
    )
    # Prefer stable source→target keys (HeteroData may not store candidate_id).
    cand_ids = [f"{s}->{t}" for s, t in zip(sources, targets)]
    return pd.DataFrame(
        {
            "candidate_id": cand_ids,
            "source": sources,
            "target": targets,
            "edge_type": edge_types,
            "label": labels,
            "p_raw": probs,
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


def score_model(hcr: str, seed: int, ckpt: Path, device: torch.device) -> dict[str, pd.DataFrame]:
    cfg = compose_cfg(hcr=hcr, seed=seed)
    train, valid, test, _ = load_recon_heterodata(cfg)
    if hcr != "none":
        enc = fit_and_attach_hcr(cfg, train, valid, test, device=device)
        if enc is not None and hasattr(cfg.model, "decoder") and cfg.model.decoder is not None:
            OmegaConf.set_struct(cfg, False)
            cfg.model.decoder.hcr_dim = int(
                getattr(train, "hcr_dim", getattr(enc.config, "output_dim", 8))
            )
            OmegaConf.set_struct(cfg, True)
    else:
        for d in (train, valid, test):
            d.hcr_enabled = False
            d.hcr_features = None
            d.hcr_supported = None

    model = load_model(cfg, train, ckpt, device)
    frames = {
        "train": predict_split(model, train, device),
        "valid": predict_split(model, valid, device),
        "test": predict_split(model, test, device),
    }
    for split, df in frames.items():
        df["edge_split"] = split
        if hcr != "none" and getattr(train if split == "train" else valid if split == "valid" else test, "hcr_supported", None) is not None:
            data = {"train": train, "valid": valid, "test": test}[split]
            df["hcr_supported"] = data.hcr_supported.detach().cpu().numpy().astype(bool)
        else:
            df["hcr_supported"] = False
    # return train message-passing edge name pairs for edge_in_train
    mp_pairs = message_passing_pairs(train)
    return {"frames": frames, "mp_pairs": mp_pairs, "train": train}


def message_passing_pairs(train_data) -> set[tuple[str, str]]:
    """Recover directed G_train edges as (src_name, dst_name) pairs."""
    pairs: set[tuple[str, str]] = set()
    name_maps = {}
    for ntype in train_data.node_types:
        store = train_data[ntype]
        if hasattr(store, "node_name"):
            name_maps[ntype] = [str(x) for x in store.node_name]
    for etype in train_data.edge_types:
        src_type, _, dst_type = etype
        edge_index = train_data[etype].edge_index
        if edge_index is None or edge_index.numel() == 0:
            continue
        src_names = name_maps.get(src_type)
        dst_names = name_maps.get(dst_type)
        if src_names is None or dst_names is None:
            continue
        for i in range(edge_index.size(1)):
            s = src_names[int(edge_index[0, i])]
            t = dst_names[int(edge_index[1, i])]
            pairs.add((s, t))
    return pairs


def calibrate_column(all_df: pd.DataFrame, raw_col: str, cal_col: str) -> pd.DataFrame:
    valid = all_df[all_df["edge_split"] == "valid"]
    calibrator = fit_platt(valid[raw_col].to_numpy(), valid["label"].to_numpy())
    out = all_df.copy()
    out[cal_col] = calibrator.transform(out[raw_col].to_numpy())
    out.attrs[f"{cal_col}_platt"] = {
        "coef": calibrator.coef,
        "intercept": calibrator.intercept,
    }
    return out


def export_seed(seed: int, out_dir: Path, device: torch.device) -> Path:
    hgt_ckpt = HGT_CKPT[seed]
    hcr2_ckpt = HCR2_CKPT[seed]
    if not hgt_ckpt.exists():
        raise FileNotFoundError(f"Missing HGT checkpoint: {hgt_ckpt}")
    if not hcr2_ckpt.exists():
        raise FileNotFoundError(f"Missing HCR2 checkpoint: {hcr2_ckpt}")

    print(f"\n=== seed {seed}: scoring HGT ===")
    hgt = score_model("none", seed, hgt_ckpt, device)
    print(f"=== seed {seed}: scoring HCR2 ===")
    hcr2 = score_model("binary_compact", seed, hcr2_ckpt, device)

    frames = []
    for split in ("train", "valid", "test"):
        a = hgt["frames"][split].rename(columns={"p_raw": "p_hgt"})
        b = hcr2["frames"][split][["candidate_id", "p_raw", "hcr_supported"]].rename(
            columns={"p_raw": "p_hcr2"}
        )
        merged = a.drop(columns=["hcr_supported"]).merge(b, on="candidate_id", how="left")
        frames.append(merged)
    df = pd.concat(frames, ignore_index=True)

    df["p_hcr3"] = np.nan
    df["p_hcr4"] = np.nan
    df = calibrate_column(df, "p_hgt", "p_hgt_calibrated")
    df = calibrate_column(df, "p_hcr2", "p_hcr2_calibrated")
    # Primary calibrated probability for graph construction: HCR-2 when available.
    df["p_calibrated"] = df["p_hcr2_calibrated"]
    df["p_raw"] = df["p_hcr2"]
    df["hcr_uncertainty"] = [
        binary_entropy_uncertainty(p) for p in df["p_hcr2_calibrated"].tolist()
    ]

    mp = hgt["mp_pairs"]
    df["edge_in_train"] = [
        (str(s), str(t)) in mp for s, t in zip(df["source"], df["target"])
    ]

    # Layer / topology from audited nodes only (not G_true edge list for scoring).
    from hcr.motifs import load_truth_graph

    cfg = compose_cfg(hcr="none", seed=seed)
    nodes, _edges = load_truth_graph(cfg)
    ranks = layer_rank_from_nodes(nodes)
    df["topological_allowed"] = annotate_topological_allowed(df, ranks).to_numpy()

    out_path = out_dir / f"seed{seed}_edges.parquet"
    try:
        df.to_parquet(out_path, index=False)
    except ImportError:
        out_path = out_dir / f"seed{seed}_edges.csv"
        df.to_csv(out_path, index=False)

    meta = {
        "seed": seed,
        "n_rows": int(len(df)),
        "hgt_checkpoint": str(hgt_ckpt),
        "hcr2_checkpoint": str(hcr2_ckpt),
        "n_train_mp_edges": int(len(mp)),
        "calibration": "platt_on_valid",
        "table_path": str(out_path),
    }
    (out_dir / f"seed{seed}_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out_path} ({len(df)} rows)")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/wave5/evidence",
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    for seed in args.seeds:
        export_seed(int(seed), args.out_dir, device)


if __name__ == "__main__":
    main()
