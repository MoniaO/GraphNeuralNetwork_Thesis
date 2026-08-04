"""Wave 7 V1 mathematical / motif-consistency checks (pre-run gates)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hcr.wave7.discrete_basis import fit_discrete_orthonormal_basis, transform_discrete_values
from hcr.wave7.encoder import Wave7HCRConfig, Wave7PairEncoder, _a11_from_binary_sample
from hcr.wave7.motif import (
    MOTIF_TYPE,
    PAIR_ROLES,
    build_candidate_coparent_triples,
    select_context,
    triples_fingerprint,
)


def test_motif_roles_are_az_ag_zg_not_aby():
    assert MOTIF_TYPE == "candidate_coparent_triangle"
    assert PAIR_ROLES == ("AZ", "AG", "ZG")
    assert "AY" not in PAIR_ROLES
    assert "BY" not in PAIR_ROLES


def test_context_selection_is_lexicographic_first():
    z, cands = select_context(("ckd", "nsaid", "acei_arb"))
    assert cands == ("acei_arb", "ckd", "nsaid")
    assert z == "acei_arb"


def test_triples_use_g_not_downstream_y():
    context_map = {("nsaid", "drug_disease_nsaid_ckd"): ("ckd",)}
    triples = build_candidate_coparent_triples(context_map)
    meta = triples[("nsaid", "drug_disease_nsaid_ckd")]
    assert meta["pairs"] == (
        ("nsaid", "ckd"),  # AZ
        ("nsaid", "drug_disease_nsaid_ckd"),  # AG
        ("ckd", "drug_disease_nsaid_ckd"),  # ZG
    )
    # Must NOT replace G with GATE_OUTCOMES child (creatinine_rise).
    assert meta["G"] == "drug_disease_nsaid_ckd"
    assert "creatinine_rise" not in {p for pair in meta["pairs"] for p in pair}


def test_v0_v1_share_identical_triples_hash():
    context_map = {
        ("nsaid", "drug_disease_nsaid_ckd"): ("ckd", "frailty"),
        ("opioid", "ddi_cns_depression_synergy"): ("benzodiazepine",),
        ("orphan", "some_gate"): (),  # empty → skipped
    }
    t0 = build_candidate_coparent_triples(context_map)
    t1 = build_candidate_coparent_triples(context_map)
    assert triples_fingerprint(t0) == triples_fingerprint(t1)
    assert PAIR_ROLES == tuple(t0[("nsaid", "drug_disease_nsaid_ckd")]["pair_roles"])
    # Selected Z is lexicographic first among candidates.
    assert t0[("nsaid", "drug_disease_nsaid_ckd")]["Z"] == "ckd"


def test_v1_support_is_n_complete_over_n_train_not_n11():
    n_train = 200
    # Rare joint positive → n11/N ≪ n_complete/N.
    u = np.zeros(n_train)
    v = np.zeros(n_train)
    u[:80] = 1.0
    v[60:140] = 1.0
    u[-10:] = np.nan
    cfg = Wave7HCRConfig(variant="W7_V1_GHCR_BINARY_ONEHOT", min_complete=50, min_level_count=5)
    enc = Wave7PairEncoder(cfg)
    enc.n_train_patients = n_train
    vec, meta = enc._transform_v1_binary_pair(u, v, u_name="x", v_name="y")
    complete = np.isfinite(u) & np.isfinite(v)
    n_complete = int(complete.sum())
    n11 = int(((u[complete] >= 0.5) & (v[complete] >= 0.5)).sum())
    assert meta["supported"] is True
    assert np.isclose(meta["support"], n_complete / n_train)
    assert not np.isclose(meta["support"], n11 / n_train)
    assert np.isclose(vec[5], n_complete / n_train)


def test_v1_rejects_sparse_levels():
    n_train = 100
    u = np.zeros(n_train)
    v = np.zeros(n_train)
    u[0] = 1.0  # only one positive
    v[:50] = 1.0
    cfg = Wave7HCRConfig(variant="W7_V1_GHCR_BINARY_ONEHOT", min_complete=50, min_level_count=5)
    enc = Wave7PairEncoder(cfg)
    enc._train_df = pd.DataFrame({"x": u, "y": v})
    enc.n_train_patients = n_train
    vec, meta = enc._transform_v1_binary_pair(u, v, u_name="x", v_name="y")
    assert meta["supported"] is False
    assert np.allclose(vec, 0.0)
    assert meta["min_level_count"] < 5


def test_v1_a11_matches_pearson_of_standardized_binary():
    rng = np.random.default_rng(1)
    n = 2000
    u = rng.integers(0, 2, size=n).astype(float)
    # Correlated binary
    v = u.copy()
    flip = rng.random(n) < 0.25
    v[flip] = 1.0 - v[flip]

    a11 = _a11_from_binary_sample(u, v, smoothing=0.0)
    p_u = u.mean()
    p_v = v.mean()
    phi_u = (u - p_u) / np.sqrt(p_u * (1 - p_u))
    phi_v = (v - p_v) / np.sqrt(p_v * (1 - p_v))
    expected = float(np.mean(phi_u * phi_v))
    assert np.isclose(a11, expected, atol=1e-10)

    # No smoothing ⇒ empirical centering / unit Gram on sample.
    cats, cons, meta = fit_discrete_orthonormal_basis(
        u, category_order=[0.0, 1.0], smoothing=0.0
    )
    assert meta["used_smoothing"] is False
    enc = transform_discrete_values(u, cats, cons)
    assert np.allclose(enc.mean(axis=0), 0.0, atol=1e-10)
    gram = enc.T @ enc / n
    assert np.allclose(gram, np.eye(1), atol=1e-10)


def test_v1_bootstrap_reports_sd_a11_not_energy_in_vector():
    rng = np.random.default_rng(2)
    n_train = 400
    u = rng.integers(0, 2, size=n_train).astype(float)
    v = u.copy()
    flip = rng.random(n_train) < 0.3
    v[flip] = 1.0 - v[flip]
    cfg = Wave7HCRConfig(
        variant="W7_V1_GHCR_BINARY_ONEHOT",
        min_complete=50,
        min_level_count=5,
        bootstrap_repeats=40,
        bootstrap_seed=0,
        v1_smoothing=0.0,
    )
    enc = Wave7PairEncoder(cfg)
    enc.n_train_patients = n_train
    vec, meta = enc._transform_v1_binary_pair(u, v, u_name="x", v_name="y")
    assert meta["supported"] is True
    assert "bootstrap_sd_a11" in meta
    assert "bootstrap_sd_energy" in meta
    assert np.isclose(vec[6], meta["bootstrap_sd_a11"])
    # Energy SD is different in general from a11 SD.
    assert meta["bootstrap_sd_a11"] >= 0.0
