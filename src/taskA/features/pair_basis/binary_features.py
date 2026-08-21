from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .bases import DEFAULT_EPS, DEFAULT_SMOOTHING, as_1d, require_finite


@dataclass(frozen=True)
class BinaryPairResult:
    p00: float
    p01: float
    p10: float
    p11: float

    p_y1_given_x1: float
    p_y1_given_x0: float

    risk_difference: float
    log_risk_ratio: float
    log_odds_ratio: float
    phi: float
    mutual_information: float

    support_x1: int
    support_y1: int
    support_joint: int
    n_samples: int


# Compact 8-d vector used by HCR-1 (binary_compact).
COMPACT_FEATURE_NAMES: tuple[str, ...] = (
    "p11",
    "conditional_probability",
    "risk_difference",
    "log_odds_ratio",
    "phi",
    "mutual_information",
    "joint_support_rate",
    "uncertainty",
)

# HCR-2: dependence + support without classical causal-effect stats.
MINIMAL_FEATURE_NAMES: tuple[str, ...] = (
    "phi",
    "mutual_information",
    "joint_support_rate",
    "uncertainty",
)

# HCR-3: classical binary association stats (non-HCR naming control).
CLASSICAL_FEATURE_NAMES: tuple[str, ...] = (
    "risk_difference",
    "log_odds_ratio",
    "joint_support_rate",
)


def _validate_binary(x: np.ndarray, name: str) -> np.ndarray:
    x = as_1d(x, name)
    x = require_finite(x, name)

    values = set(np.unique(x).tolist())
    if not values.issubset({0, 1, 0.0, 1.0}):
        raise ValueError(
            f"{name} must be binary, found values: {sorted(values)}"
        )

    return x.astype(np.int64)


def binary_pair_features(
    x: np.ndarray,
    y: np.ndarray,
    smoothing: float = DEFAULT_SMOOTHING,
    eps: float = DEFAULT_EPS,
) -> BinaryPairResult:
    """Estimate stable dependence features for a binary–binary pair.

    ``smoothing=0.5`` is a Haldane–Anscombe-style correction that keeps
    odds ratios finite when a cell is empty.
    """
    x = _validate_binary(x, "x")
    y = _validate_binary(y, "y")

    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must contain the same number of samples.")

    if smoothing < 0:
        raise ValueError(f"smoothing must be non-negative, got {smoothing}.")

    n = int(x.shape[0])

    raw_n00 = int(np.sum((x == 0) & (y == 0)))
    raw_n01 = int(np.sum((x == 0) & (y == 1)))
    raw_n10 = int(np.sum((x == 1) & (y == 0)))
    raw_n11 = int(np.sum((x == 1) & (y == 1)))

    counts = np.array(
        [
            [raw_n00, raw_n01],
            [raw_n10, raw_n11],
        ],
        dtype=np.float64,
    )

    counts_smoothed = counts + float(smoothing)
    probs = counts_smoothed / counts_smoothed.sum()

    p00, p01 = probs[0]
    p10, p11 = probs[1]

    p_x0 = p00 + p01
    p_x1 = p10 + p11
    p_y0 = p00 + p10
    p_y1 = p01 + p11

    p_y1_given_x1 = p11 / max(p_x1, eps)
    p_y1_given_x0 = p01 / max(p_x0, eps)

    risk_difference = p_y1_given_x1 - p_y1_given_x0
    risk_ratio = p_y1_given_x1 / max(p_y1_given_x0, eps)
    odds_ratio = (p11 * p00) / max(p10 * p01, eps)

    covariance = p11 - p_x1 * p_y1
    denominator = np.sqrt(p_x1 * p_x0 * p_y1 * p_y0)
    phi = covariance / max(denominator, eps)

    mutual_information = 0.0
    px = np.array([p_x0, p_x1])
    py = np.array([p_y0, p_y1])
    for i in range(2):
        for j in range(2):
            pij = probs[i, j]
            mutual_information += pij * np.log(pij / max(px[i] * py[j], eps))

    return BinaryPairResult(
        p00=float(p00),
        p01=float(p01),
        p10=float(p10),
        p11=float(p11),
        p_y1_given_x1=float(p_y1_given_x1),
        p_y1_given_x0=float(p_y1_given_x0),
        risk_difference=float(risk_difference),
        log_risk_ratio=float(np.log(max(risk_ratio, eps))),
        log_odds_ratio=float(np.log(max(odds_ratio, eps))),
        phi=float(phi),
        mutual_information=float(mutual_information),
        support_x1=int(raw_n10 + raw_n11),
        support_y1=int(raw_n01 + raw_n11),
        support_joint=int(raw_n11),
        n_samples=int(n),
    )


def _feature_dict(result: BinaryPairResult) -> dict[str, float]:
    joint_support_rate = result.support_joint / max(result.n_samples, 1)
    uncertainty = 1.0 / np.sqrt(result.support_joint + 1.0)
    return {
        "p11": float(result.p11),
        "conditional_probability": float(result.p_y1_given_x1),
        "risk_difference": float(result.risk_difference),
        "log_risk_ratio": float(result.log_risk_ratio),
        "log_odds_ratio": float(result.log_odds_ratio),
        "phi": float(result.phi),
        "mutual_information": float(result.mutual_information),
        "joint_support_rate": float(joint_support_rate),
        "uncertainty": float(uncertainty),
    }


def result_to_vector(
    result: BinaryPairResult,
    feature_names: Sequence[str] | None = None,
) -> np.ndarray:
    """Map a BinaryPairResult to a compact float32 feature vector."""
    names: Iterable[str] = feature_names or COMPACT_FEATURE_NAMES
    values = _feature_dict(result)
    missing = [name for name in names if name not in values]
    if missing:
        raise KeyError(f"Unknown HCR feature names: {missing}")
    return np.array([values[name] for name in names], dtype=np.float32)
