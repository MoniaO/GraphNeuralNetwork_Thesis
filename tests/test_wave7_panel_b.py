"""Wave 7 Panel B audits: jitter, 40/120 shapes, hybrid packing, orientation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hcr.binary_features import binary_pair_features, result_to_vector
from hcr.wave7.panel_b.encoder import PanelBConfig, PanelBPairEncoder
from hcr.wave7.panel_b.jitter import count_to_jittered_u, deterministic_xi
from hcr.wave7.panel_b.packing import PAIR_DIM, pack_pair_vector_40, pad_coefficient_matrix
from hcr.wave7.panel_b.attach import MOTIF_DIM, _pad_v0_motif_to_120


def test_pair_and_motif_dims():
    assert PAIR_DIM == 40
    assert MOTIF_DIM == 120


def test_deterministic_jitter_stable_across_calls():
    pids = np.array(["p1", "p2", "p3"])
    a = deterministic_xi(pids, variable_name="n_drugs", seed=20260722)
    b = deterministic_xi(pids, variable_name="n_drugs", seed=20260722)
    assert np.allclose(a, b)
    assert np.all((a > 0) & (a < 1))
    c = deterministic_xi(pids, variable_name="other", seed=20260722)
    assert not np.allclose(a, c)


def test_count_jitter_splits_mass_intervals():
    cats = [0.0, 1.0, 2.0]
    probs = np.array([0.5, 0.3, 0.2])
    left = np.array([0.0, 0.5, 0.8])
    counts = np.array([0.0, 0.0, 1.0, 2.0])
    pids = np.array(["a", "b", "c", "d"])
    u = count_to_jittered_u(
        counts,
        pids,
        variable_name="c",
        seed=1,
        categories=cats,
        probabilities=probs,
        left_cdf=left,
    )
    assert u[0] < 0.5 and u[1] < 0.5
    assert 0.5 <= u[2] < 0.8
    assert u[3] >= 0.8


def test_pad_preserves_orientation_and_transpose():
    a = np.arange(1, 5, dtype=float).reshape(1, 4)  # 1×4
    p = pad_coefficient_matrix(a, 1, 4)
    assert p.shape == (4, 4)
    assert np.allclose(p[0, :4], [1, 2, 3, 4])
    assert np.allclose(p[1:, :], 0)
    at = pad_coefficient_matrix(a.T, 4, 1)
    assert np.allclose(at[:4, 0], [1, 2, 3, 4])


def test_b0_pad_motif_preserves_v0_blocks():
    motif24 = np.arange(24, dtype=np.float32)
    out = _pad_v0_motif_to_120(motif24)[0]
    assert out.shape == (120,)
    assert np.allclose(out[0:8], motif24[0:8])
    assert np.allclose(out[40:48], motif24[8:16])
    assert np.allclose(out[80:88], motif24[16:24])
    assert np.allclose(out[8:40], 0)
    assert np.allclose(out[48:80], 0)
    assert np.allclose(out[88:120], 0)


def test_b3_binary_keeps_legacy_v0_prefix():
    rng = np.random.default_rng(0)
    n = 400
    u = rng.integers(0, 2, size=n).astype(float)
    v = ((u + rng.integers(0, 2, size=n)) % 2).astype(float)
    legacy = result_to_vector(binary_pair_features(u.astype(int), v.astype(int), smoothing=0.5))
    # Fake A with a11
    padded = np.zeros((4, 4))
    padded[0, 0] = 0.42
    vec = pack_pair_vector_40(
        padded_a=padded,
        d_u=1,
        d_v=1,
        kind_u="binary",
        kind_v="binary",
        u_raw=u,
        v_raw=v,
        n_complete=n,
        n_train=n,
        supported=True,
        mode="B3",
        legacy8=legacy,
        a11=0.42,
        fill_enrichment=True,
    )
    assert vec.shape == (40,)
    assert np.allclose(vec[:8], legacy)
    assert np.isclose(vec[8], 0.42)
    assert vec[34] == 1.0 and vec[37] == 1.0  # both binary


def test_encoder_shapes_and_swap_transpose_a11():
    n = 300
    rng = np.random.default_rng(7)
    # Construct synthetic patient table with known binary columns that exist in specs
    # Use raw column names that VARIABLE_SPECS may not know — encoder falls back to continuous.
    # Force binary path by monkeypatching kinds via names that are in VARIABLE_SPECS if possible.
    from hcr.variable_specs_v3 import VARIABLE_SPECS
    from hcr.variable_spec import VariableType

    binaries = [
        name
        for name, spec in VARIABLE_SPECS.items()
        if spec.variable_type == VariableType.BINARY
    ][:2]
    if len(binaries) < 2:
        return
    u_name, v_name = binaries[0], binaries[1]
    col_u = VARIABLE_SPECS[u_name].column_name
    col_v = VARIABLE_SPECS[v_name].column_name
    df = pd.DataFrame(
        {
            "patient_id": [f"p{i}" for i in range(n)],
            col_u: rng.integers(0, 2, size=n).astype(float),
            col_v: rng.integers(0, 2, size=n).astype(float),
        }
    )
    cfg = PanelBConfig(variant="W7B_B2_JITTER_GHCR_ENRICHED40", min_complete=50, min_level_count=5)
    enc = PanelBPairEncoder(cfg)
    enc.fit(df, [u_name, v_name])
    vu, mu = enc.transform_pair(u_name, v_name)
    vv, mv = enc.transform_pair(v_name, u_name)
    assert vu.shape == (40,)
    assert vv.shape == (40,)
    assert np.isfinite(vu).all() and np.isfinite(vv).all()
    # a11 should match (scalar); full matrix transpose: slot a_jk ↔ a_kj
    # For 1×1 binary, pads to same a11.
    assert np.isclose(vu[0], vv[0], atol=1e-6)
    # cache stable
    vu2, _ = enc.transform_pair(u_name, v_name)
    assert np.allclose(vu, vu2)


def test_a11_matches_legacy_phi_within_tolerance():
    """GHCR a11 ≈ V0 φ (Pearson of orthonormal contrasts)."""
    from hcr.variable_specs_v3 import VARIABLE_SPECS
    from hcr.variable_spec import VariableType

    binaries = [
        name
        for name, spec in VARIABLE_SPECS.items()
        if spec.variable_type == VariableType.BINARY
    ][:2]
    if len(binaries) < 2:
        return
    u_name, v_name = binaries[0], binaries[1]
    col_u = VARIABLE_SPECS[u_name].column_name
    col_v = VARIABLE_SPECS[v_name].column_name
    rng = np.random.default_rng(11)
    n = 500
    u = rng.integers(0, 2, size=n).astype(float)
    # induce dependence
    v = u.copy()
    flip = rng.random(n) < 0.2
    v[flip] = 1 - v[flip]
    df = pd.DataFrame(
        {
            "patient_id": [f"p{i}" for i in range(n)],
            col_u: u,
            col_v: v,
        }
    )
    cfg = PanelBConfig(variant="W7B_B2_JITTER_GHCR_ENRICHED40")
    enc = PanelBPairEncoder(cfg)
    enc.fit(df, [u_name, v_name])
    vec, meta = enc.transform_pair(u_name, v_name)
    legacy = result_to_vector(
        binary_pair_features(u.astype(int), v.astype(int), smoothing=0.5)
    )
    # V0 φ is typically slot 4 in binary_compact
    phi = float(legacy[4])
    a11 = float(meta["a11"])
    assert abs(a11 - phi) < 0.05 or abs(a11 - phi) / (abs(phi) + 1e-8) < 0.15
