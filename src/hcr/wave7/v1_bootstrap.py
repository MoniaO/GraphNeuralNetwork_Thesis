"""V1 binary GHCR bootstrap: reference loop + vectorized production path.

Mathematical procedure (identical in both implementations):
  resample I_UV with replacement → recompute p_U,p_V → rebuild contrasts → a11
  report SD(a11), not SD(a11²).

For smoothing=0:
  p = n1 / n
which matches the closed-form standardized-binary a11 used in the point estimate.
"""

from __future__ import annotations

import hashlib

import numpy as np


def canonical_pair_names(u_name: str, v_name: str) -> tuple[str, str]:
    a, b = str(u_name), str(v_name)
    return (a, b) if a <= b else (b, a)


def v1_bootstrap_seed(
    *,
    global_seed: int,
    scenario: str,
    u_name: str,
    v_name: str,
) -> int:
    """Deterministic seed independent of pair iteration / cache order."""
    left, right = canonical_pair_names(u_name, v_name)
    payload = f"{int(global_seed)}|{scenario}|{left}|{right}|v1_bootstrap_a11"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**63 - 1)


def _a11_binary_with_smoothing(u: np.ndarray, v: np.ndarray, *, smoothing: float) -> float:
    """Single-sample a11 with optional additive smoothing on binary margins."""
    uu = np.asarray(u, dtype=np.float64)
    vv = np.asarray(v, dtype=np.float64)
    n = float(len(uu))
    if n < 2:
        return 0.0
    s = float(smoothing)
    n1_u = float(uu.sum())
    n1_v = float(vv.sum())
    p_u = (n1_u + s) / (n + 2.0 * s)
    p_v = (n1_v + s) / (n + 2.0 * s)
    if not (0.0 < p_u < 1.0 and 0.0 < p_v < 1.0):
        return 0.0
    denom_u = np.sqrt(p_u * (1.0 - p_u))
    denom_v = np.sqrt(p_v * (1.0 - p_v))
    if denom_u <= 1e-12 or denom_v <= 1e-12:
        return 0.0
    phi_u = (uu - p_u) / denom_u
    phi_v = (vv - p_v) / denom_v
    return float(np.mean(phi_u * phi_v))


def _bootstrap_sd_a11_reference(
    u: np.ndarray,
    v: np.ndarray,
    *,
    repeats: int,
    smoothing: float,
    seed: int,
) -> tuple[float, int]:
    """Private reference loop — used only by regression tests."""
    sd_a11, _sd_energy, n_valid = _bootstrap_a11_stats_reference(
        u, v, repeats=repeats, smoothing=smoothing, seed=seed
    )
    return sd_a11, n_valid


def _bootstrap_a11_stats_reference(
    u: np.ndarray,
    v: np.ndarray,
    *,
    repeats: int,
    smoothing: float,
    seed: int,
) -> tuple[float, float, int]:
    uu = np.asarray(u, dtype=np.float64)
    vv = np.asarray(v, dtype=np.float64)
    n = len(uu)
    rng = np.random.default_rng(int(seed))
    boots: list[float] = []
    for _ in range(int(repeats)):
        idx = rng.integers(0, n, size=n, endpoint=False)
        u_b, v_b = uu[idx], vv[idx]
        if u_b.min() == u_b.max() or v_b.min() == v_b.max():
            continue
        boots.append(_a11_binary_with_smoothing(u_b, v_b, smoothing=smoothing))
    if len(boots) < 2:
        return 0.0, 0.0, len(boots)
    arr = np.asarray(boots, dtype=np.float64)
    return float(np.std(arr, ddof=1)), float(np.std(arr**2, ddof=1)), len(boots)


def _bootstrap_sd_a11_vectorized(
    u: np.ndarray,
    v: np.ndarray,
    *,
    repeats: int,
    smoothing: float,
    seed: int,
    chunk_size: int = 10,
) -> tuple[float, int]:
    sd_a11, _sd_energy, n_valid = _bootstrap_a11_stats_vectorized(
        u, v, repeats=repeats, smoothing=smoothing, seed=seed, chunk_size=chunk_size
    )
    return sd_a11, n_valid


def _bootstrap_a11_stats_vectorized(
    u: np.ndarray,
    v: np.ndarray,
    *,
    repeats: int,
    smoothing: float,
    seed: int,
    chunk_size: int = 10,
) -> tuple[float, float, int]:
    """Production vectorized bootstrap; same RNG stream as the reference loop."""
    uu = np.asarray(u, dtype=np.float64)
    vv = np.asarray(v, dtype=np.float64)
    n = len(uu)
    repeats = int(repeats)
    if n < 2 or repeats < 1:
        return 0.0, 0.0, 0

    rng = np.random.default_rng(int(seed))
    index_rows = [rng.integers(0, n, size=n, endpoint=False) for _ in range(repeats)]
    indices = np.stack(index_rows, axis=0)

    s = float(smoothing)
    boots: list[float] = []
    for start in range(0, repeats, max(int(chunk_size), 1)):
        block = indices[start : start + chunk_size]
        u_boot = uu[block]
        v_boot = vv[block]
        n1_u = u_boot.sum(axis=1)
        n1_v = v_boot.sum(axis=1)
        p_u = (n1_u + s) / (n + 2.0 * s)
        p_v = (n1_v + s) / (n + 2.0 * s)
        denom_u = np.sqrt(p_u * (1.0 - p_u))
        denom_v = np.sqrt(p_v * (1.0 - p_v))
        level_ok = (u_boot.min(axis=1) < u_boot.max(axis=1)) & (
            v_boot.min(axis=1) < v_boot.max(axis=1)
        )
        valid = (
            level_ok
            & np.isfinite(denom_u)
            & np.isfinite(denom_v)
            & (denom_u > 1e-12)
            & (denom_v > 1e-12)
        )
        if not np.any(valid):
            continue
        phi_u = (u_boot[valid] - p_u[valid, None]) / denom_u[valid, None]
        phi_v = (v_boot[valid] - p_v[valid, None]) / denom_v[valid, None]
        a11_boot = np.mean(phi_u * phi_v, axis=1)
        boots.extend(float(x) for x in a11_boot.tolist())

    if len(boots) < 2:
        return 0.0, 0.0, len(boots)
    arr = np.asarray(boots, dtype=np.float64)
    return float(np.std(arr, ddof=1)), float(np.std(arr**2, ddof=1)), len(boots)
