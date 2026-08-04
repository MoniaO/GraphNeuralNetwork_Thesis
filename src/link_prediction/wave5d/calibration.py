"""Validation-only temperature / intercept calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class Calibrator:
    temperature: float = 1.0
    intercept: float = 0.0

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        return (np.asarray(logits, dtype=np.float64) - self.intercept) / max(
            self.temperature, 1e-6
        )

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        z = self.transform_logits(logits)
        return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


def fit_temperature_bias(
    logits: np.ndarray,
    labels: np.ndarray,
) -> Calibrator:
    """Fit P = σ((ℓ - b) / τ) on validation edges via 1-d logistic regression."""
    logits = np.asarray(logits, dtype=np.float64).reshape(-1, 1)
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    if np.unique(labels).size < 2:
        return Calibrator()
    clf = LogisticRegression(solver="lbfgs", max_iter=1000)
    clf.fit(logits, labels)
    # σ(a*ℓ + b) ≡ σ((ℓ - (-b/a)) / (1/a)) with a>0 preferred.
    a = float(clf.coef_.ravel()[0])
    b = float(clf.intercept_.ravel()[0])
    if abs(a) < 1e-8:
        return Calibrator()
    temperature = 1.0 / a
    intercept = -b / a
    if temperature < 0:
        temperature = abs(temperature)
        intercept = -intercept
    return Calibrator(temperature=float(temperature), intercept=float(intercept))


def fit_platt(logits: np.ndarray, labels: np.ndarray) -> Calibrator:
    return fit_temperature_bias(logits, labels)


def bce_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, labels)
