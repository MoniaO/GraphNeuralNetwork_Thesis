#!/usr/bin/env python3
"""Wave 5A.1 control — patient_shuffle_refit evidence.

Permute each patient-feature column independently, then refit HCR pairwise
features, score with the frozen HCR-2 decoder, and recalibrate.

HGT scores are taken from the already-exported Wave-5 evidence tables
(structural model unchanged). Only the HCR branch is destroyed.
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
from data.candidate_pairs import candidate_pairs_from_data, unique_pairs
from data.patient_matrix import load_patient_matrix_with_split, train_patient_df
from hcr.pair_encoder import HCRPairEncoder
from hcr.variable_specs_v3 import VARIABLE_SPECS
from models.TaskA.hetero_gnn import HeteroReconGNN
from wnerw.calibration import binary_entropy_uncertainty, fit_platt
from wnerw.graph_builder import annotate_topological_allowed, layer_rank_from_nodes

SEEDS = (20260721, 20260722, 20260723, 20260724, 20260725)

HCR2_CKPT = {
    20260721: ROOT / "outputs/2026-07-31/15-48-14/best_model.pt",
    20260722: ROOT / "outputs/2026-07-31/15-52-24/best_model.pt",
    20260723: ROOT / "outputs/2026-07-31/15-57-05/best_model.pt",
    20260724: ROOT / "outputs/2026-07-31/16-16-19/best_model.pt",
    20260725: ROOT / "outputs/2026-07-31/16-18-42/best_model.pt",
}


def compose_cfg(seed: int):
    overrides = [
        "model=TaskA_hgt_hcr",
        "hcr=binary_compact",
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
        "experiment.wave=WAVE5A1_PATIENT_SHUFFLE",
        "experiment.motif_completion.enabled=false",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name="config", overrides=overrides)


def shuffle_patient_columns(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    skip = {"patient_id", "split", "patient_split", "id"}
    for col in out.columns:
        if col in skip:
            continue
        if not pd.api.types.is_numeric_dtype(out[col]) and out[col].dtype != object:
            continue
        # Permute values independently — destroys within-patient dependence.
        out[col] = rng.permutation(out[col].to_numpy())
    return out


@torch.no_grad()
def predict_split(model, data, device) -> pd.DataFrame:
    model.eval()
    data = data.to(device)
    probs = torch.sigmoid(model(data).view(-1)).cpu().numpy()
    labels = data.edge_label.cpu().numpy().astype(np.float64).reshape(-1)
    sources = list(data.candidate_source_name)
    targets = list(data.candidate_target_name)
    return pd.DataFrame(
        {
            "candidate_id": [f"{s}->{t}" for s, t in zip(sources, targets)],
            "source": sources,
            "target": targets,
            "label": labels,
            "p_hcr2_shuffled": probs,
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


def export_seed(seed: int, base_evidence: Path, out_dir: Path, device: torch.device) -> Path:
    base = pd.read_csv(base_evidence) if base_evidence.suffix == ".csv" else pd.read_parquet(base_evidence)
    cfg = compose_cfg(seed)
    train, valid, test, _ = load_recon_heterodata(cfg)

    # Fit HCR on column-shuffled patients.
    rng = np.random.default_rng(int(seed) + 17)
    patient_df = load_patient_matrix_with_split(cfg)
    patient_df = shuffle_patient_columns(patient_df, rng)
    train_patients = train_patient_df(patient_df)

    encoder = HCRPairEncoder.from_hydra(cfg.hcr, VARIABLE_SPECS)
    all_pairs = unique_pairs(
        candidate_pairs_from_data(train)
        + candidate_pairs_from_data(valid)
        + candidate_pairs_from_data(test)
    )
    encoder.fit(train_patient_df=train_patients, candidate_pairs=all_pairs)
    for data in (train, valid, test):
        pairs = candidate_pairs_from_data(data)
        data.hcr_features = encoder.transform(pairs, device=device)
        data.hcr_supported = torch.as_tensor(
            encoder.supported_mask(pairs), dtype=torch.bool, device=device
        )
        data.hcr_enabled = True
        data.hcr_dim = int(encoder.config.output_dim)

    if hasattr(cfg.model, "decoder") and cfg.model.decoder is not None:
        OmegaConf.set_struct(cfg, False)
        cfg.model.decoder.hcr_dim = int(getattr(train, "hcr_dim", 8))
        OmegaConf.set_struct(cfg, True)

    model = load_model(cfg, train, HCR2_CKPT[seed], device)
    frames = []
    for split, data in (("train", train), ("valid", valid), ("test", test)):
        fr = predict_split(model, data, device)
        fr["edge_split"] = split
        fr["hcr_supported"] = data.hcr_supported.detach().cpu().numpy().astype(bool)
        frames.append(fr)
    shuf = pd.concat(frames, ignore_index=True)

    # Keep HGT / graph flags from base evidence; replace HCR branch.
    keep_cols = [
        c
        for c in base.columns
        if c
        not in {
            "p_hcr2",
            "p_hcr2_calibrated",
            "p_calibrated",
            "p_raw",
            "hcr_uncertainty",
            "hcr_supported",
        }
    ]
    df = base[keep_cols].merge(
        shuf[["candidate_id", "p_hcr2_shuffled", "hcr_supported"]],
        on="candidate_id",
        how="inner",
    )
    df = df.rename(columns={"p_hcr2_shuffled": "p_hcr2"})

    valid = df[df["edge_split"] == "valid"]
    cal = fit_platt(valid["p_hcr2"].to_numpy(), valid["label"].to_numpy())
    df["p_hcr2_calibrated"] = cal.transform(df["p_hcr2"].to_numpy())
    df["p_calibrated"] = df["p_hcr2_calibrated"]
    df["p_raw"] = df["p_hcr2"]
    df["hcr_uncertainty"] = [
        binary_entropy_uncertainty(p) for p in df["p_hcr2_calibrated"].tolist()
    ]

    # Refresh topological_allowed annotation (candidate-only flag).
    from hcr.motifs import load_truth_graph

    nodes, _ = load_truth_graph(cfg)
    ranks = layer_rank_from_nodes(nodes)
    df["topological_allowed"] = annotate_topological_allowed(df, ranks).to_numpy()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"seed{seed}_patient_shuffle.csv"
    df.to_csv(out_path, index=False)
    meta = {
        "seed": seed,
        "control": "patient_shuffle_refit",
        "n_rows": int(len(df)),
        "base_evidence": str(base_evidence),
        "hcr2_checkpoint": str(HCR2_CKPT[seed]),
        "note": "patient feature columns permuted before HCR fit+score; HGT kept",
    }
    (out_dir / f"seed{seed}_patient_shuffle_meta.json").write_text(
        json.dumps(meta, indent=2)
    )
    print(f"Wrote {out_path} ({len(df)} rows)")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-evidence-dir",
        type=Path,
        default=ROOT / "outputs/wave5/evidence",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/wave5/evidence_patient_shuffle",
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    for seed in args.seeds:
        base = args.base_evidence_dir / f"seed{seed}_edges.csv"
        if not base.exists():
            base = args.base_evidence_dir / f"seed{seed}_edges.parquet"
        if not base.exists():
            raise FileNotFoundError(base)
        export_seed(int(seed), base, args.out_dir, device)


if __name__ == "__main__":
    main()
