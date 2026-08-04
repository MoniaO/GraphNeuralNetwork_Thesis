"""Probability calibration for WNERW edge evidence (fit on validation only)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class PlattCalibrator:
    """Platt scaling: ``p_cal = σ(a · logit(p_raw) + b)`` fitted on validation."""

    coef: float
    intercept: float
    epsilon: float = 1e-6

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        logits = _safe_logit(np.asarray(probabilities, dtype=np.float64), self.epsilon)
        z = self.coef * logits + self.intercept
        return 1.0 / (1.0 + np.exp(-z))


@dataclass(frozen=True)
class TemperatureCalibrator:
    """Temperature scaling on logits: ``p_cal = σ(logit(p_raw) / T)``."""

    temperature: float
    epsilon: float = 1e-6

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        logits = _safe_logit(np.asarray(probabilities, dtype=np.float64), self.epsilon)
        z = logits / self.temperature
        return 1.0 / (1.0 + np.exp(-z))


def _safe_logit(p: np.ndarray, epsilon: float) -> np.ndarray:
    p = np.clip(p, epsilon, 1.0 - epsilon)
    return np.log(p) - np.log1p(-p)


def fit_platt(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> PlattCalibrator:
    """Fit Platt scaling on validation probabilities / binary labels."""
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if probabilities.shape != labels.shape:
        raise ValueError("probabilities and labels must share shape")
    if len(np.unique(labels)) < 2:
        # Degenerate split — identity map in logit space.
        return PlattCalibrator(coef=1.0, intercept=0.0, epsilon=epsilon)

    logits = _safe_logit(probabilities, epsilon).reshape(-1, 1)
    clf = LogisticRegression(solver="lbfgs", max_iter=1000)
    clf.fit(logits, labels)
    return PlattCalibrator(
        coef=float(clf.coef_.ravel()[0]),
        intercept=float(clf.intercept_.ravel()[0]),
        epsilon=epsilon,
    )


def fit_temperature(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    grid: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0),
    epsilon: float = 1e-6,
) -> TemperatureCalibrator:
    """Pick temperature on a small grid by minimizing validation NLL."""
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    logits = _safe_logit(probabilities, epsilon)

    best_t = 1.0
    best_nll = float("inf")
    for temperature in grid:
        z = logits / temperature
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, epsilon, 1.0 - epsilon)
        nll = -float(np.mean(labels * np.log(p) + (1.0 - labels) * np.log(1.0 - p)))
        if nll < best_nll:
            best_nll = nll
            best_t = float(temperature)
    return TemperatureCalibrator(temperature=best_t, epsilon=epsilon)


def binary_entropy_uncertainty(probability: float, epsilon: float = 1e-6) -> float:
    """Normalized binary entropy in [0, 1] as a simple uncertainty proxy."""
    p = min(max(float(probability), epsilon), 1.0 - epsilon)
    h = -p * math_log2(p) - (1.0 - p) * math_log2(1.0 - p)
    return float(h)  # already in [0, 1] for base-2 entropy of a Bernoulli


def math_log2(x: float) -> float:
    return float(np.log(x) / np.log(2.0))
