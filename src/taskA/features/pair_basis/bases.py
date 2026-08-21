"""Shared numeric helpers for HCR feature estimators."""

from __future__ import annotations

import numpy as np


DEFAULT_EPS = 1e-12
DEFAULT_SMOOTHING = 0.5


def as_1d(array: np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(array).reshape(-1)
    if out.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return out


def require_finite(array: np.ndarray, name: str) -> np.ndarray:
    if np.isnan(array).any() or np.isinf(array).any():
        raise ValueError(f"{name} contains non-finite values.")
    return array
