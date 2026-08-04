from __future__ import annotations

import numpy as np
import pytest

from hcr.binary_features import binary_pair_features, result_to_vector


def test_perfect_positive_association():
    x = np.array([0, 0, 1, 1], dtype=np.int64)
    y = np.array([0, 0, 1, 1], dtype=np.int64)
    result = binary_pair_features(x, y, smoothing=0.5)
    assert result.n_samples == 4
    assert result.support_joint == 2
    assert result.risk_difference > 0
    assert result.phi > 0
    assert result.log_odds_ratio > 0
    vector = result_to_vector(result)
    assert vector.shape == (8,)
    assert np.isfinite(vector).all()


def test_independence_near_zero_phi():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, size=20000)
    y = rng.integers(0, 2, size=20000)
    result = binary_pair_features(x, y, smoothing=0.5)
    assert abs(result.phi) < 0.05
    assert abs(result.risk_difference) < 0.05


def test_rejects_non_binary():
    with pytest.raises(ValueError, match="binary"):
        binary_pair_features(np.array([0, 1, 2]), np.array([0, 1, 0]))


def test_empty_joint_cell_finite_or():
    x = np.array([0, 0, 1, 1], dtype=np.int64)
    y = np.array([0, 1, 0, 0], dtype=np.int64)  # n11 = 0
    result = binary_pair_features(x, y, smoothing=0.5)
    assert result.support_joint == 0
    assert np.isfinite(result.log_odds_ratio)
