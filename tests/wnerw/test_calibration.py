"""Calibration helpers."""

from __future__ import annotations

import numpy as np

from wnerw.calibration import fit_platt, fit_temperature


def test_platt_improves_or_stays_finite():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=200)
    labels = (logits > 0).astype(int)
    # Mis-calibrated raw probs (overconfident)
    raw = 1.0 / (1.0 + np.exp(-3.0 * logits))
    cal = fit_platt(raw, labels)
    out = cal.transform(raw)
    assert np.all((out > 0) & (out < 1))
    assert np.isfinite(out).all()


def test_temperature_grid_returns_positive_t():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, size=100)
    y = (p > 0.5).astype(float)
    t = fit_temperature(p, y)
    assert t.temperature > 0
