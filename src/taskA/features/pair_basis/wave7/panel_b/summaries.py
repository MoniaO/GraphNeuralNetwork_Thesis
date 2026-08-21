"""Marginal and joint-activity summaries for Panel B 40-d pair vectors."""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def _safe_entropy_norm(probs: np.ndarray, *, log_base_k: float) -> float:
    p = np.asarray(probs, dtype=float)
    p = p[p > 0]
    if p.size == 0 or log_base_k <= 0:
        return 0.0
    h = -float(np.sum(p * np.log(p)))
    return float(np.clip(h / log_base_k, 0.0, 1.0))


def binary_marginal(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = max(len(x), 1)
    p1 = float(np.mean(x >= 0.5)) if len(x) else 0.0
    p1c = float(np.clip(p1, EPS, 1.0 - EPS))
    h = -(p1c * np.log(p1c) + (1 - p1c) * np.log(1 - p1c)) / np.log(2.0)
    logit = np.tanh(np.log(p1c / (1 - p1c)) / 4.0)
    n_pos = int(np.sum(x >= 0.5))
    rarity = 1.0 / np.sqrt(n_pos + 1.0)
    return np.array([p1, float(h), float(logit), float(rarity)], dtype=np.float64)


def count_marginal(values: np.ndarray, *, count_cap: float) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.zeros(4, dtype=np.float64)
    xc = np.minimum(x, float(count_cap))
    p0 = float(np.mean(xc <= 0))
    mu_log = float(np.mean(np.log1p(xc)) / max(np.log1p(count_cap), EPS))
    cats, counts = np.unique(xc, return_counts=True)
    probs = counts / counts.sum()
    h_norm = _safe_entropy_norm(probs, log_base_k=np.log(max(len(cats), 2)))
    mean = float(np.mean(xc))
    var = float(np.var(xc))
    dispersion = float(np.tanh(np.log((var + EPS) / (mean + EPS))))
    return np.array([p0, mu_log, h_norm, dispersion], dtype=np.float64)


def continuous_marginal(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (4-d summary, robust_z for all finite-aligned input with nan elsewhere)."""
    x = np.asarray(values, dtype=float)
    out_z = np.full(len(x), np.nan, dtype=np.float64)
    finite = np.isfinite(x)
    if finite.sum() < 2:
        return np.zeros(4, dtype=np.float64), out_z
    xf = x[finite]
    med = float(np.median(xf))
    mad = float(np.median(np.abs(xf - med)))
    scale = 1.4826 * mad + EPS
    zr = (xf - med) / scale
    out_z[finite] = zr
    # skew / excess kurtosis
    m = zr.mean()
    s = zr.std(ddof=0) + EPS
    z0 = (zr - m) / s
    skew = float(np.mean(z0**3))
    kurt = float(np.mean(z0**4) - 3.0)
    extreme = float(np.mean(np.abs(zr) > 1.5))
    imbalance = float(np.mean(zr > 1.5) - np.mean(zr < -1.5))
    summary = np.array(
        [np.tanh(skew / 3.0), np.tanh(kurt / 10.0), extreme, imbalance],
        dtype=np.float64,
    )
    return summary, out_z


def active_mask(values: np.ndarray, kind: str, *, robust_z: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if kind == "binary":
        return np.isfinite(x) & (x >= 0.5)
    if kind == "count":
        return np.isfinite(x) & (x > 0)
    assert robust_z is not None
    return np.isfinite(robust_z) & (np.abs(robust_z) > 1.5)


def joint_activity_slots(
    u: np.ndarray,
    v: np.ndarray,
    *,
    kind_u: str,
    kind_v: str,
    robust_z_u: np.ndarray | None,
    robust_z_v: np.ndarray | None,
) -> np.ndarray:
    complete = np.isfinite(u) & np.isfinite(v)
    if complete.sum() == 0:
        return np.zeros(4, dtype=np.float64)
    au = active_mask(u, kind_u, robust_z=robust_z_u)
    av = active_mask(v, kind_v, robust_z=robust_z_v)
    both = complete & au & av
    n = int(complete.sum())
    p_u = float(np.sum(complete & au) / n)
    p_v = float(np.sum(complete & av) / n)
    p_pp = float(np.sum(both) / n)
    n_pp = int(np.sum(both))
    delta = p_pp - p_u * p_v
    loglift = float(np.log((p_pp + EPS) / (p_u * p_v + EPS)))
    rarity = 1.0 / np.sqrt(n_pp + 1.0)
    return np.array([p_pp, delta, loglift, rarity], dtype=np.float64)


def type_onehot(kind: str) -> np.ndarray:
    out = np.zeros(3, dtype=np.float64)
    if kind == "binary":
        out[0] = 1.0
    elif kind == "count":
        out[1] = 1.0
    else:
        out[2] = 1.0
    return out
