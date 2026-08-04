#!/usr/bin/env python3
"""Supported vs unsupported candidate metrics for Wave-3 HCR.

Loads frozen FINAL HGT checkpoints and WAVE3 binary_compact checkpoints,
scores valid/test candidates split by binary–binary support, and writes
paired Δ tables without retraining.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from sklearn.metrics import average_precision_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.PreprocessingTaskA.load_hetero_recon_data import load_recon_heterodata
from hcr.attach import fit_and_attach_hcr
from hcr.variable_spec import VariableType
from hcr.variable_specs_v3 import VARIABLE_SPECS
from models.TaskA.hetero_gnn import HeteroReconGNN


def _find_checkpoints() -> dict[tuple[str, int], Path]:
    """Map (kind, seed) → checkpoint. kind in {baseline, compact}."""
    out: dict[tuple[str, int], Path] = {}
    for summary_path in sorted((ROOT / "wandb").glob("run-*/files/wandb-summary.json")):
        summary = json.loads(summary_path.read_text())
        seed = summary.get("training_seed")
        ckpt = summary.get("checkpoint_path")
        if seed is None or not ckpt:
            continue
        path = Path(str(ckpt))
        if not path.exists():
            continue
        exp = str(summary.get("experiment_name") or "")
        intervention = str(summary.get("intervention") or "")
        hcr = str(summary.get("hcr_variant") or "")
        encoder = str(summary.get("encoder_name") or "")
        heads = summary.get("heads")

        if exp == "ARCH_FINAL" and encoder == "hgt" and int(heads or 0) == 8:
            out[("baseline", int(seed))] = path
        if exp == "WAVE3_HCR" and hcr == "binary_compact":
            out[("compact", int(seed))] = path
        if "hgt_hcr_binary_compact" in intervention and "SMOKE" not in exp:
            out[("compact", int(seed))] = path
    return out


def _compose_cfg(seed: int, hcr_enabled: bool):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        overrides = [
            "model=TaskA_hgt_hcr" if hcr_enabled else "model=TaskA_hgt",
            "hcr=binary_compact" if hcr_enabled else "hcr=none",
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
        ]
        return compose(config_name="config", overrides=overrides)


def _supported_mask(data) -> np.ndarray:
    sources = list(data.candidate_source_name)
    targets = list(data.candidate_target_name)
    mask = []
    for s, t in zip(sources, targets):
        ss = VARIABLE_SPECS.get(s)
        ts = VARIABLE_SPECS.get(t)
        ok = (
            ss is not None
            and ts is not None
            and ss.variable_type == VariableType.BINARY
            and ts.variable_type == VariableType.BINARY
        )
        mask.append(bool(ok))
    return np.asarray(mask, dtype=bool)


@torch.no_grad()
def _score(model, data, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    data = data.to(device)
    logits = model(data).view(-1)
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    labels = data.edge_label.detach().cpu().numpy().astype(np.float64).reshape(-1)
    return labels, probs


def _subset_metrics(labels: np.ndarray, probs: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    y = labels[mask]
    p = probs[mask]
    n = int(y.size)
    n_pos = int(y.sum()) if n else 0
    if n == 0 or n_pos == 0 or n_pos == n:
        return {
            "n": n,
            "n_pos": n_pos,
            "auprc": float("nan"),
            "brier": float("nan"),
        }
    return {
        "n": n,
        "n_pos": n_pos,
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def _load_model(cfg, train_data, ckpt_path: Path, device) -> HeteroReconGNN:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = HeteroReconGNN(
        cfg=cfg,
        data=train_data,
        in_channels=int(train_data[train_data.node_types[0]].x.size(-1)),
        hidden_channels=int(getattr(cfg.model, "hidden_channels", 64)),
    ).to(device)
    # Materialize lazy params if any
    with torch.no_grad():
        _ = model(train_data.to(device))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", default=None)
    parser.add_argument("--seeds", default="20260721,20260722,20260723")
    args = parser.parse_args()
    day = args.day or date.today().isoformat()
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    out = ROOT / "outputs" / f"wave3_hcr_{day}"
    out.mkdir(parents=True, exist_ok=True)

    ckpts = _find_checkpoints()
    rows = []

    # Data/layout identical across seeds; HCR features also seed-independent.
    cfg_hcr = _compose_cfg(seeds[0], hcr_enabled=True)
    train_data, valid_data, test_data, _ = load_recon_heterodata(cfg_hcr)
    fit_and_attach_hcr(cfg_hcr, train_data, valid_data, test_data, device="cpu")
    cfg_base = _compose_cfg(seeds[0], hcr_enabled=False)
    # Baseline uses same graphs/candidates but no hcr_features required.
    for data in (train_data, valid_data, test_data):
        # keep hcr tensors for compact; baseline model ignores them
        pass

    valid_mask = _supported_mask(valid_data)
    test_mask = _supported_mask(test_data)
    print(
        f"valid supported={valid_mask.mean():.3f} "
        f"({valid_mask.sum()}/{len(valid_mask)}); "
        f"test supported={test_mask.mean():.3f} "
        f"({test_mask.sum()}/{len(test_mask)})"
    )

    device = torch.device("cpu")
    for seed in seeds:
        base_ckpt = ckpts.get(("baseline", seed))
        compact_ckpt = ckpts.get(("compact", seed))
        if base_ckpt is None or compact_ckpt is None:
            print(f"SKIP seed={seed}: missing ckpt baseline={base_ckpt} compact={compact_ckpt}")
            continue

        cfg_b = _compose_cfg(seed, hcr_enabled=False)
        cfg_c = _compose_cfg(seed, hcr_enabled=True)
        # Align hcr_dim with attached features
        if hasattr(cfg_c.model, "decoder"):
            OmegaConf.set_struct(cfg_c, False)
            cfg_c.model.decoder.hcr_dim = int(getattr(train_data, "hcr_dim", 8))
            OmegaConf.set_struct(cfg_c, True)

        model_b = _load_model(cfg_b, train_data, base_ckpt, device)
        model_c = _load_model(cfg_c, train_data, compact_ckpt, device)

        for split, data, mask in (
            ("valid", valid_data, valid_mask),
            ("test", test_data, test_mask),
        ):
            y_b, p_b = _score(model_b, data, device)
            y_c, p_c = _score(model_c, data, device)
            for support_name, m in (("supported", mask), ("unsupported", ~mask), ("all", np.ones_like(mask, dtype=bool))):
                mb = _subset_metrics(y_b, p_b, m)
                mc = _subset_metrics(y_c, p_c, m)
                rows.append(
                    {
                        "training_seed": seed,
                        "split": split,
                        "subset": support_name,
                        "n": mb["n"],
                        "n_pos": mb["n_pos"],
                        "auprc_baseline": mb["auprc"],
                        "auprc_compact": mc["auprc"],
                        "delta_auprc": (
                            mc["auprc"] - mb["auprc"]
                            if np.isfinite(mb["auprc"]) and np.isfinite(mc["auprc"])
                            else float("nan")
                        ),
                        "brier_baseline": mb["brier"],
                        "brier_compact": mc["brier"],
                        "delta_brier": (
                            mc["brier"] - mb["brier"]
                            if np.isfinite(mb["brier"]) and np.isfinite(mc["brier"])
                            else float("nan")
                        ),
                        "baseline_ckpt": str(base_ckpt),
                        "compact_ckpt": str(compact_ckpt),
                    }
                )
        print(f"scored seed={seed}")

    frame = pd.DataFrame(rows)
    detail = out / f"{day}_wave3_HCR_supported_vs_unsupported.csv"
    frame.to_csv(detail, index=False)

    if not frame.empty:
        summary = (
            frame.groupby(["split", "subset"], dropna=False)
            .agg(
                n_seeds=("training_seed", "nunique"),
                mean_delta_auprc=("delta_auprc", "mean"),
                std_delta_auprc=("delta_auprc", "std"),
                mean_auprc_baseline=("auprc_baseline", "mean"),
                mean_auprc_compact=("auprc_compact", "mean"),
                mean_delta_brier=("delta_brier", "mean"),
            )
            .reset_index()
        )
        summary_path = out / f"{day}_wave3_HCR_supported_vs_unsupported_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(summary.to_string(index=False))
        print(f"Wrote {detail}")
        print(f"Wrote {summary_path}")
    else:
        print("No rows scored")


if __name__ == "__main__":
    main()
