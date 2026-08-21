"""Pack Panel B ordered-pair features into fixed 40-d vectors."""

from __future__ import annotations

import numpy as np

from taskA.features.pair_basis.binary_features import binary_pair_features, result_to_vector

from .summaries import (
    EPS,
    binary_marginal,
    continuous_marginal,
    count_marginal,
    joint_activity_slots,
    type_onehot,
)

PAIR_DIM = 40
MATRIX_SLOTS = 16


def pad_coefficient_matrix(matrix: np.ndarray, d_u: int, d_v: int) -> np.ndarray:
    """Pad A (d_u × d_v) to 4×4 preserving orientation (zeros elsewhere)."""
    a = np.asarray(matrix, dtype=np.float64)
    out = np.zeros((4, 4), dtype=np.float64)
    if a.size == 0:
        return out
    r = min(int(d_u), 4, a.shape[0] if a.ndim == 2 else 0)
    c = min(int(d_v), 4, a.shape[1] if a.ndim == 2 else 0)
    if r and c:
        out[:r, :c] = a[:r, :c]
    return out


def dependence_summaries(padded: np.ndarray, d_u: int, d_v: int) -> np.ndarray:
    a = np.asarray(padded, dtype=np.float64)
    total = float(np.sum(a**2))
    denom = max(int(d_u) * int(d_v), 1)
    mean_e = total / denom
    low = float(np.sum(a[:2, :2] ** 2))
    frac = low / (total + EPS)
    max_abs = float(np.max(np.abs(a))) if a.size else 0.0
    return np.array([total, mean_e, frac, max_abs], dtype=np.float64)


def legacy_binary_compact_8(u: np.ndarray, v: np.ndarray, *, smoothing: float = 0.5) -> np.ndarray:
    mask = np.isfinite(u) & np.isfinite(v)
    if mask.sum() < 2:
        return np.zeros(8, dtype=np.float32)
    uu = np.clip(np.round(u[mask]), 0, 1).astype(np.int64)
    vv = np.clip(np.round(v[mask]), 0, 1).astype(np.int64)
    if not (set(np.unique(uu)).issubset({0, 1}) and set(np.unique(vv)).issubset({0, 1})):
        return np.zeros(8, dtype=np.float32)
    result = binary_pair_features(uu, vv, smoothing=float(smoothing))
    return result_to_vector(result).astype(np.float32)


def binary_ghcr_extras(u: np.ndarray, v: np.ndarray, a11: float) -> np.ndarray:
    """Slots 8–15 for hybrid B3 binary–binary."""
    mask = np.isfinite(u) & np.isfinite(v)
    uu = np.clip(np.round(u[mask]), 0, 1).astype(np.float64)
    vv = np.clip(np.round(v[mask]), 0, 1).astype(np.float64)
    if len(uu) == 0:
        return np.zeros(8, dtype=np.float64)
    p_u = float(uu.mean())
    p_v = float(vv.mean())
    p11 = float(np.mean((uu >= 0.5) & (vv >= 0.5)))
    hu = binary_marginal(uu)[1]
    hv = binary_marginal(vv)[1]
    delta = p11 - p_u * p_v
    loglift = float(np.log((p11 + EPS) / (p_u * p_v + EPS)))
    return np.array(
        [a11, a11**2, p_u, p_v, hu, hv, delta, loglift], dtype=np.float64
    )


def pack_pair_vector_40(
    *,
    padded_a: np.ndarray,
    d_u: int,
    d_v: int,
    kind_u: str,
    kind_v: str,
    u_raw: np.ndarray,
    v_raw: np.ndarray,
    n_complete: int,
    n_train: int,
    supported: bool,
    mode: str,
    legacy8: np.ndarray | None = None,
    a11: float = 0.0,
    count_cap_u: float = 1.0,
    count_cap_v: float = 1.0,
    fill_enrichment: bool = True,
    fill_matrix: bool = True,
) -> np.ndarray:
    """Build the 40-d Panel B pair vector.

    mode:
      B0 — legacy8 in 0..7, rest zero (caller may pass only binary)
      B1 — matrix + dependence summaries only
      B2/B3 — full enrichment; B3 binary uses legacy+extras via legacy8/a11
    """
    out = np.zeros(PAIR_DIM, dtype=np.float64)
    padded = pad_coefficient_matrix(padded_a, d_u, d_v)

    if mode == "B0":
        # Width-control only: legacy V0 in 0..7, strict zero-pad thereafter.
        if legacy8 is not None:
            out[:8] = np.asarray(legacy8, dtype=np.float64).reshape(-1)[:8]
        return out.astype(np.float32)

    if mode == "B3" and kind_u == "binary" and kind_v == "binary" and legacy8 is not None:
        out[:8] = np.asarray(legacy8, dtype=np.float64).reshape(-1)[:8]
        out[8:16] = binary_ghcr_extras(u_raw, v_raw, a11)
    elif fill_matrix:
        out[:16] = padded.reshape(-1, order="C")

    if fill_matrix:
        out[16:20] = dependence_summaries(padded, d_u=max(d_u, 1), d_v=max(d_v, 1))

    if fill_enrichment:
        # Marginals
        rz_u = rz_v = None
        if kind_u == "binary":
            out[20:24] = binary_marginal(u_raw)
        elif kind_u == "count":
            out[20:24] = count_marginal(u_raw, count_cap=count_cap_u)
        else:
            out[20:24], rz_u = continuous_marginal(u_raw)

        if kind_v == "binary":
            out[24:28] = binary_marginal(v_raw)
        elif kind_v == "count":
            out[24:28] = count_marginal(v_raw, count_cap=count_cap_v)
        else:
            out[24:28], rz_v = continuous_marginal(v_raw)

        # Need robust z for continuous activity even if we already computed summary
        if kind_u == "continuous" and rz_u is None:
            _, rz_u = continuous_marginal(u_raw)
        if kind_v == "continuous" and rz_v is None:
            _, rz_v = continuous_marginal(v_raw)

        out[28:32] = joint_activity_slots(
            u_raw,
            v_raw,
            kind_u=kind_u,
            kind_v=kind_v,
            robust_z_u=rz_u,
            robust_z_v=rz_v,
        )

    out[32] = float(n_complete / max(n_train, 1))
    out[33] = 1.0 if supported else 0.0
    out[34:37] = type_onehot(kind_u)
    out[37:40] = type_onehot(kind_v)
    return out.astype(np.float32)
