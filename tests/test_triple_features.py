from __future__ import annotations

import numpy as np
import pytest

from hcr.triple_features import (
    binary_triple_features,
    shuffled_z_triple_features,
    triple_result_to_vector,
)


def test_pure_and_gate_has_positive_a111():
    # X∧Z → Y almost surely when both on
    rng = np.random.default_rng(0)
    n = 8000
    x = rng.integers(0, 2, size=n)
    z = rng.integers(0, 2, size=n)
    y = ((x == 1) & (z == 1)).astype(np.int64)
    # add a little noise
    flip = rng.random(n) < 0.02
    y = np.where(flip, 1 - y, y)
    result = binary_triple_features(x, y, z)
    assert result.a111_excess > 0.15
    assert result.p_y1_given_x1z1 > 0.9
    vec = triple_result_to_vector(result)
    assert vec.shape == (17,)
    assert np.isfinite(vec).all()


def test_independent_triple_near_zero_a111():
    rng = np.random.default_rng(1)
    n = 20000
    x = rng.integers(0, 2, size=n)
    y = rng.integers(0, 2, size=n)
    z = rng.integers(0, 2, size=n)
    result = binary_triple_features(x, y, z)
    assert abs(result.a111_excess) < 0.05


def test_shuffle_destroys_interaction():
    rng = np.random.default_rng(2)
    n = 10000
    x = rng.integers(0, 2, size=n)
    z = rng.integers(0, 2, size=n)
    y = ((x == 1) & (z == 1)).astype(np.int64)
    true = binary_triple_features(x, y, z)
    shuf = shuffled_z_triple_features(x, y, z, seed=99)
    assert true.a111_excess > shuf.a111_excess + 0.1


def test_without_a111_ablation_zeros_slot():
    x = np.array([0, 0, 1, 1])
    z = np.array([0, 1, 0, 1])
    y = np.array([0, 0, 0, 1])
    result = binary_triple_features(x, y, z)
    full = triple_result_to_vector(result, include_a111=True)
    ablated = triple_result_to_vector(result, include_a111=False)
    assert ablated[9] == 0.0 and ablated[10] == 0.0
    assert np.allclose(full[:9], ablated[:9])
    assert np.allclose(full[11:], ablated[11:])
