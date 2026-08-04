from __future__ import annotations

import numpy as np
import pytest

from hcr.wave7.discrete_basis import (
    audit_basis,
    fit_discrete_orthonormal_basis,
    transform_discrete_values,
)
from hcr.wave7.mapping import pack_pair_vector, select_explicit_coefficients


def test_binary_one_contrast_orthonormal():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, size=5000).astype(float)
    categories, contrasts, meta = fit_discrete_orthonormal_basis(x, smoothing=0.5)
    assert meta["K"] == 2
    assert contrasts.shape == (2, 1)
    encoded = transform_discrete_values(x, categories, contrasts)
    report = audit_basis(encoded, mean_atol=1e-2, gram_atol=5e-2)
    assert report["ok_mean"]
    assert report["ok_gram"]


def test_count_k_minus_1_and_fail_above_max():
    values = np.array([0, 1, 2, 0, 1, 2, 3, 3], dtype=float)
    categories, contrasts, meta = fit_discrete_orthonormal_basis(
        values, max_categories=12
    )
    assert meta["K"] == 4
    assert contrasts.shape == (4, 3)
    encoded = transform_discrete_values(values, categories, contrasts)
    report = audit_basis(encoded, mean_atol=1e-6, gram_atol=1e-4)
    assert report["ok"]

    many = np.arange(21, dtype=float)
    with pytest.raises(ValueError, match="one-hot limit"):
        fit_discrete_orthonormal_basis(many, max_categories=20)


def test_unknown_category_maps_to_zero():
    train = np.array([0.0, 1.0, 2.0, 0.0, 1.0])
    categories, contrasts, _ = fit_discrete_orthonormal_basis(train)
    out = transform_discrete_values(np.array([9.0]), categories, contrasts)
    assert np.allclose(out, 0.0)


def test_pack_pair_vector_binary_count_energy_uses_all():
    # 1 x 6 coefficient row → only first 4 explicit, energy uses all 6
    matrix = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
    c = select_explicit_coefficients(matrix, "binary_count")
    assert np.allclose(c, [0.1, 0.2, 0.3, 0.4])
    vec = pack_pair_vector(
        matrix,
        kind_u="binary",
        kind_v="count",
        support=0.9,
        bootstrap_sd=0.01,
        supported=True,
    )
    assert vec.shape == (8,)
    assert np.isclose(vec[4], np.sum(matrix**2))
    assert vec[7] == 1.0
