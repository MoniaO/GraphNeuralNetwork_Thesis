from __future__ import annotations

import numpy as np
import pandas as pd


def abs_spearman(x, y) -> float:
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    value = x.corr(y, method="spearman")
    return 0.0 if pd.isna(value) else abs(float(value))


def _rank_uniform(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").rank(method="average", pct=True).to_numpy(float)


def _rbf(values: np.ndarray) -> np.ndarray:
    values = values.reshape(-1, 1)
    distances = (values - values.T) ** 2
    positive = distances[distances > 0]
    sigma2 = max(float(np.median(positive)) if positive.size else 1.0, 1e-8)
    return np.exp(-distances / (2.0 * sigma2))


def normalized_hsic(x, y, *, max_n: int = 2000, seed: int = 20260722) -> float:
    """One nonlinear comparator, deliberately capped because HSIC is O(n^2)."""
    frame = pd.DataFrame({"x": x, "y": y}).apply(pd.to_numeric, errors="coerce").dropna()
    if len(frame) < 20:
        return float("nan")
    if len(frame) > max_n:
        frame = frame.sample(max_n, random_state=seed)
    xr, yr = _rank_uniform(frame["x"]), _rank_uniform(frame["y"])
    k, l = _rbf(xr), _rbf(yr)
    n = len(frame)
    h = np.eye(n) - np.ones((n, n)) / n
    kc, lc = h @ k @ h, h @ l @ h
    numerator = float(np.sum(kc * lc))
    denominator = np.sqrt(float(np.sum(kc * kc) * np.sum(lc * lc)))
    return numerator / denominator if denominator > 0 else 0.0
