"""Third-order binary HCR features for Wave 3B.

Distinguishes:
  - pairwise marginal dependence (HCR-2 style)
  - pure triple interaction a111 = E[φx φz φy], φ=2X-1
  - conditional dependence of (X,Y) given Z
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bases import DEFAULT_EPS, DEFAULT_SMOOTHING, as_1d, require_finite
from .binary_features import (
    BinaryPairResult,
    binary_pair_features,
    result_to_vector,
)


@dataclass(frozen=True)
class BinaryTripleResult:
    # Pairwise blocks
    pair_xy: BinaryPairResult
    pair_xz: BinaryPairResult
    pair_zy: BinaryPairResult

    # Pure third-order interaction (Rademacher)
    a111: float
    a111_excess: float
    p_y1_given_x1z1: float

    # Conditional dependence of (x,y) | z
    phi_xy_given_z0: float
    phi_xy_given_z1: float
    delta_phi_z: float
    rd_xy_given_z0: float
    rd_xy_given_z1: float
    delta_rd_z: float
    conditional_mi_xy_given_z: float
    confounding_shift: float  # phi_marginal - E_z[phi|z]

    support_z0: int
    support_z1: int
    support_xyz_111: int
    n_samples: int


HCR3_FEATURE_NAMES: tuple[str, ...] = (
    # pair xy compact subset
    "xy_p11",
    "xy_rd",
    "xy_log_or",
    "xy_phi",
    "xy_mi",
    # pair xz / zy phi+mi
    "xz_phi",
    "xz_mi",
    "zy_phi",
    "zy_mi",
    # third-order / conditional
    "a111_excess",
    "p_y1_given_x1z1",
    "conditional_mi_xy_given_z",
    "delta_rd_z",
    "delta_phi_z",
    "confounding_shift",
    "support_xyz_111_rate",
    "uncertainty_triple",
)


def _validate_binary(x: np.ndarray, name: str) -> np.ndarray:
    x = as_1d(x, name)
    x = require_finite(x, name)
    values = set(np.unique(x).tolist())
    if not values.issubset({0, 1, 0.0, 1.0}):
        raise ValueError(f"{name} must be binary, found {sorted(values)}")
    return x.astype(np.int64)


def _phi_from_counts(n00: float, n01: float, n10: float, n11: float, eps: float) -> float:
    counts = np.array([[n00, n01], [n10, n11]], dtype=np.float64) + DEFAULT_SMOOTHING
    probs = counts / counts.sum()
    p00, p01 = probs[0]
    p10, p11 = probs[1]
    p_x1 = p10 + p11
    p_x0 = p00 + p01
    p_y1 = p01 + p11
    p_y0 = p00 + p10
    cov = p11 - p_x1 * p_y1
    denom = np.sqrt(max(p_x1 * p_x0 * p_y1 * p_y0, eps))
    return float(cov / denom)


def _rd_from_counts(n00: float, n01: float, n10: float, n11: float, eps: float) -> float:
    counts = np.array([[n00, n01], [n10, n11]], dtype=np.float64) + DEFAULT_SMOOTHING
    probs = counts / counts.sum()
    p00, p01 = probs[0]
    p10, p11 = probs[1]
    p_x1 = p10 + p11
    p_x0 = p00 + p01
    return float(p11 / max(p_x1, eps) - p01 / max(p_x0, eps))


def _mi_2x2(probs: np.ndarray, eps: float) -> float:
    px = probs.sum(axis=1)
    py = probs.sum(axis=0)
    mi = 0.0
    for i in range(2):
        for j in range(2):
            pij = probs[i, j]
            mi += pij * np.log(pij / max(px[i] * py[j], eps))
    return float(mi)


def binary_triple_features(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    smoothing: float = DEFAULT_SMOOTHING,
    eps: float = DEFAULT_EPS,
) -> BinaryTripleResult:
    """Estimate pairwise + interaction + conditional features for (X,Y,Z).

    Convention for motif / latent-gate experiments:
      x = source / component A
      y = target / outcome or gate
      z = co-parent / conditioner
    """
    x = _validate_binary(x, "x")
    y = _validate_binary(y, "y")
    z = _validate_binary(z, "z")
    if not (x.shape == y.shape == z.shape):
        raise ValueError("x, y, z must have the same length")

    n = int(x.shape[0])
    pair_xy = binary_pair_features(x, y, smoothing=smoothing, eps=eps)
    pair_xz = binary_pair_features(x, z, smoothing=smoothing, eps=eps)
    pair_zy = binary_pair_features(z, y, smoothing=smoothing, eps=eps)

    # Rademacher interaction
    phi_x = 2.0 * x.astype(np.float64) - 1.0
    phi_y = 2.0 * y.astype(np.float64) - 1.0
    phi_z = 2.0 * z.astype(np.float64) - 1.0
    a111 = float(np.mean(phi_x * phi_z * phi_y))
    # For rare AND gates, raw a111 is dominated by the (0,0,0) cell.
    # Excess over independence isolates the synergistic mass.
    a111_indep = float(np.mean(phi_x) * np.mean(phi_z) * np.mean(phi_y))
    a111_excess = float(a111 - a111_indep)
    # Direct AND activation probability (clinically readable).
    p_y1_given_x1z1 = float(
        np.mean(y[(x == 1) & (z == 1)]) if np.any((x == 1) & (z == 1)) else 0.0
    )
    support_xyz_111 = int(np.sum((x == 1) & (y == 1) & (z == 1)))
    support_z0 = int(np.sum(z == 0))
    support_z1 = int(np.sum(z == 1))

    # Conditional 2x2 tables for (x,y)|z
    def slice_counts(zv: int) -> tuple[float, float, float, float]:
        m = z == zv
        xx, yy = x[m], y[m]
        n00 = float(np.sum((xx == 0) & (yy == 0)))
        n01 = float(np.sum((xx == 0) & (yy == 1)))
        n10 = float(np.sum((xx == 1) & (yy == 0)))
        n11 = float(np.sum((xx == 1) & (yy == 1)))
        return n00, n01, n10, n11

    c0 = slice_counts(0)
    c1 = slice_counts(1)
    phi0 = _phi_from_counts(*c0, eps=eps)
    phi1 = _phi_from_counts(*c1, eps=eps)
    rd0 = _rd_from_counts(*c0, eps=eps)
    rd1 = _rd_from_counts(*c1, eps=eps)

    # Conditional MI ≈ P(z) MI(x,y|z)
    def cond_mi(counts: tuple[float, float, float, float], pz: float) -> float:
        arr = np.array([[counts[0], counts[1]], [counts[2], counts[3]]], dtype=np.float64)
        arr = arr + smoothing
        probs = arr / arr.sum()
        return pz * _mi_2x2(probs, eps)

    pz0 = support_z0 / max(n, 1)
    pz1 = support_z1 / max(n, 1)
    cmi = cond_mi(c0, pz0) + cond_mi(c1, pz1)

    phi_marginal = pair_xy.phi
    phi_cond_expectation = pz0 * phi0 + pz1 * phi1
    confounding_shift = float(phi_marginal - phi_cond_expectation)

    return BinaryTripleResult(
        pair_xy=pair_xy,
        pair_xz=pair_xz,
        pair_zy=pair_zy,
        a111=a111,
        a111_excess=a111_excess,
        p_y1_given_x1z1=p_y1_given_x1z1,
        phi_xy_given_z0=phi0,
        phi_xy_given_z1=phi1,
        delta_phi_z=float(phi1 - phi0),
        rd_xy_given_z0=rd0,
        rd_xy_given_z1=rd1,
        delta_rd_z=float(rd1 - rd0),
        conditional_mi_xy_given_z=float(cmi),
        confounding_shift=confounding_shift,
        support_z0=support_z0,
        support_z1=support_z1,
        support_xyz_111=support_xyz_111,
        n_samples=n,
    )


def triple_result_to_vector(
    result: BinaryTripleResult,
    include_a111: bool = True,
) -> np.ndarray:
    """Compact HCR-3 vector (17-d). Optionally drop triple-interaction slots."""
    xy = result.pair_xy
    xz = result.pair_xz
    zy = result.pair_zy
    support_rate = result.support_xyz_111 / max(result.n_samples, 1)
    uncertainty = 1.0 / np.sqrt(result.support_xyz_111 + 1.0)
    a_ex = float(result.a111_excess) if include_a111 else 0.0
    p_and = float(result.p_y1_given_x1z1) if include_a111 else 0.0
    values = [
        xy.p11,
        xy.risk_difference,
        xy.log_odds_ratio,
        xy.phi,
        xy.mutual_information,
        xz.phi,
        xz.mutual_information,
        zy.phi,
        zy.mutual_information,
        a_ex,
        p_and,
        result.conditional_mi_xy_given_z,
        result.delta_rd_z,
        result.delta_phi_z,
        result.confounding_shift,
        support_rate,
        uncertainty,
    ]
    return np.asarray(values, dtype=np.float32)


def shuffled_z_triple_features(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    rng: np.random.Generator | None = None,
    seed: int = 20260722,
) -> BinaryTripleResult:
    """Placebo: permute Z so third-order structure is destroyed."""
    rng = rng or np.random.default_rng(seed)
    z_shuf = np.asarray(z).copy()
    rng.shuffle(z_shuf)
    return binary_triple_features(x, y, z_shuf)
