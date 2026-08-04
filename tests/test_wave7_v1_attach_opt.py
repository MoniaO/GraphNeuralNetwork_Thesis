"""Regression: optimized V1 bootstrap/cache vs reference loop."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hcr.variable_spec import VariableType
from hcr.variable_specs_v3 import VARIABLE_SPECS
from hcr.wave7.encoder import Wave7HCRConfig, Wave7PairEncoder, _a11_from_binary_sample
from hcr.wave7.v1_bootstrap import (
    _bootstrap_sd_a11_reference,
    _bootstrap_sd_a11_vectorized,
    canonical_pair_names,
    v1_bootstrap_seed,
)


def _binary_var_names(limit: int = 40) -> list[str]:
    names = [
        n
        for n, spec in VARIABLE_SPECS.items()
        if spec.variable_type == VariableType.BINARY
    ]
    return sorted(names)[:limit]


def test_vectorized_bootstrap_matches_reference_loop():
    rng = np.random.default_rng(0)
    for i in range(20):
        n = 800
        u = rng.integers(0, 2, size=n).astype(np.float64)
        v = u.copy()
        flip = rng.random(n) < 0.35
        v[flip] = 1.0 - v[flip]
        seed = 10_000 + i
        sd_ref, n_ref = _bootstrap_sd_a11_reference(
            u, v, repeats=30, smoothing=0.0, seed=seed
        )
        sd_vec, n_vec = _bootstrap_sd_a11_vectorized(
            u, v, repeats=30, smoothing=0.0, seed=seed
        )
        assert n_ref == n_vec
        assert abs(sd_ref - sd_vec) <= 1e-10


def test_v1_pair_math_and_symmetry_and_cache():
    rng = np.random.default_rng(1)
    n_patients = 1200
    names = _binary_var_names(8)
    assert len(names) >= 4
    data = {
        name: rng.integers(0, 2, size=n_patients).astype(float) for name in names
    }
    # correlate a few
    data[names[1]] = data[names[0]].copy()
    flip = rng.random(n_patients) < 0.2
    data[names[1]][flip] = 1.0 - data[names[1]][flip]
    df = pd.DataFrame(data)

    cfg = Wave7HCRConfig(
        variant="W7_V1_GHCR_BINARY_ONEHOT",
        bootstrap_repeats=30,
        v1_smoothing=0.0,
        min_complete=50,
        min_level_count=5,
        bootstrap_seed=20260722,
    )
    enc = Wave7PairEncoder(cfg)
    enc.scenario = "clean"
    enc.fit(df, names)

    pairs = [(names[i], names[j]) for i in range(4) for j in range(i + 1, 5)]
    assert len(pairs) >= 20 // 2  # at least several
    # pad to 20 by resampling correlated draws
    while len(pairs) < 20:
        pairs.append(pairs[len(pairs) % len(pairs)])

    checked = 0
    for u_name, v_name in pairs[:20]:
        u = pd.to_numeric(df[enc._column(u_name)], errors="coerce").to_numpy()
        v = pd.to_numeric(df[enc._column(v_name)], errors="coerce").to_numpy()
        complete = np.isfinite(u) & np.isfinite(v)
        uu = np.clip(np.round(u[complete]), 0, 1).astype(np.float64)
        vv = np.clip(np.round(v[complete]), 0, 1).astype(np.float64)
        if len(uu) < 50:
            continue
        a11 = _a11_from_binary_sample(uu, vv, smoothing=0.0)
        seed = v1_bootstrap_seed(
            global_seed=20260722, scenario="clean", u_name=u_name, v_name=v_name
        )
        sd_ref, _ = _bootstrap_sd_a11_reference(
            uu, vv, repeats=30, smoothing=0.0, seed=seed
        )
        vec, meta = enc.transform_pair(u_name, v_name)
        assert vec.shape == (8,)
        assert np.isfinite(vec).all()
        # Exact math in float64 metadata; float32 vector has storage rounding only.
        assert abs(float(meta["a11"]) - a11) <= 1e-12
        assert abs(float(meta["energy"]) - a11**2) <= 1e-12
        assert abs(float(vec[0]) - a11) <= 1e-6
        assert abs(float(vec[4]) - a11**2) <= 1e-6
        assert float(vec[5]) == pytest.approx(len(uu) / n_patients)
        assert abs(float(meta["bootstrap_sd_a11"]) - sd_ref) <= 1e-10
        assert abs(float(vec[6]) - sd_ref) <= 1e-6
        assert float(vec[7]) == 1.0
        assert meta["fit_scope"] == "patient_train_only"
        assert "valid" not in meta["patient_train_fingerprint"]
        assert "test" not in meta["patient_train_fingerprint"]

        vec_rev, meta_rev = enc.transform_pair(v_name, u_name)
        assert np.allclose(vec, vec_rev, atol=1e-12)
        assert meta_rev.get("cache_hit") is True
        checked += 1

    assert checked >= 10


def test_canonical_pair_key_order_independent():
    assert canonical_pair_names("b", "a") == ("a", "b")
    s1 = v1_bootstrap_seed(global_seed=1, scenario="clean", u_name="a", v_name="b")
    s2 = v1_bootstrap_seed(global_seed=1, scenario="clean", u_name="b", v_name="a")
    assert s1 == s2
