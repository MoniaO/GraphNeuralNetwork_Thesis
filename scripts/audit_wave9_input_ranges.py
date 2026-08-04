#!/usr/bin/env python3
"""Pre-flight: export ranges for all 40 B2 pair slots (train-fit only).

Writes: outputs/wave9/audits/input_feature_ranges.csv

Does NOT fit scalers on valid/test. Uses the same Wave7C B2 attach path
as the frozen A1 baseline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "wave9" / "audits"
SEED = 20260722


def _slot_stats(col: np.ndarray) -> dict:
    x = col.astype(np.float64)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return {
            "min": np.nan,
            "p01": np.nan,
            "p05": np.nan,
            "median": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "fraction_zero": np.nan,
            "fraction_outside_-3_3": np.nan,
            "n": 0,
        }
    return {
        "min": float(np.min(finite)),
        "p01": float(np.quantile(finite, 0.01)),
        "p05": float(np.quantile(finite, 0.05)),
        "median": float(np.median(finite)),
        "p95": float(np.quantile(finite, 0.95)),
        "p99": float(np.quantile(finite, 0.99)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "fraction_zero": float(np.mean(np.abs(finite) < 1e-12)),
        "fraction_outside_-3_3": float(np.mean((finite < -3.0) | (finite > 3.0))),
        "n": int(finite.size),
    }


def main() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from data.PreprocessingTaskA.load_hetero_recon_data import load_recon_heterodata
    from hcr.wave7.wave7c.attach import fit_and_attach_wave7c

    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "model=TaskA_hgt_wave7c",
                "hcr=w7c_b2_audit",
                "data.dataset.scenario=clean",
                "data.feature_ablation_profile=empirical",
                f"data.candidate_seed={SEED}",
                f"training.seed={SEED}",
                "experiment.wave=WAVE9",
                "experiment.variant=W9_K0_MLP_CONTROL",
                "wandb.enabled=false",
            ],
        )

    train, valid, test, _ = load_recon_heterodata(cfg)
    fit_and_attach_wave7c(cfg, train, valid, test, device="cpu")

    # Pool all role blocks from train candidates (AZ/AG/ZG × n_candidates).
    feats = train.hcr_features.detach().cpu().numpy()  # [N, 120]
    blocks = feats.reshape(feats.shape[0], 3, 40)
    # Flatten roles: each row is one 40D pair vector used by a role encoder.
    pairs = blocks.reshape(-1, 40)

    rows = []
    for s in range(40):
        st = _slot_stats(pairs[:, s])
        st["slot"] = s
        st["n_pair_vectors"] = int(pairs.shape[0])
        rows.append(st)

    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    cols = [
        "slot",
        "min",
        "p01",
        "p05",
        "median",
        "p95",
        "p99",
        "max",
        "mean",
        "std",
        "fraction_zero",
        "fraction_outside_-3_3",
        "n",
        "n_pair_vectors",
    ]
    path = OUT / "input_feature_ranges.csv"
    df[cols].to_csv(path, index=False)

    summary = {
        "seed": SEED,
        "scenario": "clean",
        "n_candidates_train": int(feats.shape[0]),
        "n_pair_vectors": int(pairs.shape[0]),
        "slots_with_gt_5pct_outside_grid": int(
            (df["fraction_outside_-3_3"] > 0.05).sum()
        ),
        "path": str(path.relative_to(ROOT)),
        "note": (
            "If many slots exceed grid [-3,3], consider train-only scaler "
            "applied identically to K0 and K1, then re-reproduce K0."
        ),
    }
    (OUT / "input_feature_ranges_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
