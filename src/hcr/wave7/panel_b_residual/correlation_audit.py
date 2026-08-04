"""Pre-train correlation audit: V0 8-d vs B2 40-d on binary–binary pairs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from hcr.binary_features import COMPACT_FEATURE_NAMES


def write_v0_vs_b2_correlations(
    *,
    v0_vectors: np.ndarray,
    b2_vectors: np.ndarray,
    out_path: Path,
) -> pd.DataFrame:
    """Pool BB pairs; correlate every V0 feature with every B2 feature."""
    v0 = np.asarray(v0_vectors, dtype=np.float64)
    b2 = np.asarray(b2_vectors, dtype=np.float64)
    if v0.ndim != 2 or b2.ndim != 2 or v0.shape[0] != b2.shape[0]:
        raise ValueError(f"bad shapes v0={v0.shape} b2={b2.shape}")
    if v0.shape[1] != 8 or b2.shape[1] != 40:
        raise ValueError(f"expected V0=8 B2=40, got {v0.shape[1]} / {b2.shape[1]}")

    rows = []
    for i, v0_name in enumerate(COMPACT_FEATURE_NAMES):
        x = v0[:, i]
        if not np.isfinite(x).all() or np.std(x) < 1e-12:
            continue
        for j in range(b2.shape[1]):
            y = b2[:, j]
            if not np.isfinite(y).all() or np.std(y) < 1e-12:
                pearson = float("nan")
                spearman = float("nan")
            else:
                pearson = float(np.corrcoef(x, y)[0, 1])
                spearman = float(spearmanr(x, y).correlation)
            abs_corr = abs(pearson) if np.isfinite(pearson) else float("nan")
            rows.append(
                {
                    "v0_feature": v0_name,
                    "v0_slot": i,
                    "b2_feature": f"b2_{j}",
                    "b2_slot": j,
                    "pearson_corr": pearson,
                    "spearman_corr": spearman,
                    "abs_corr": abs_corr,
                    "cluster_gt_0_95": bool(np.isfinite(abs_corr) and abs_corr > 0.95),
                    "n_pairs": int(v0.shape[0]),
                }
            )
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df
