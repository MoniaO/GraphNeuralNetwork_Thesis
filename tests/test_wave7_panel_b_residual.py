"""Audits for Wave 7 Panel B residual fusion + conditional edge gain."""

from __future__ import annotations

import numpy as np
import torch

from hcr.wave7.panel_b_residual.conditional_edge_gain import (
    compute_conditional_edge_gain,
)
from hcr.wave7.panel_b_residual.constants import (
    GATE_SLOTS,
    MOTIF_B2_DIM,
    MOTIF_CLASSICAL4_DIM,
    MOTIF_LATENT_WITH_TRIPLE,
    MOTIF_V0_DIM,
    PAIR_LATENT_DIM,
    TRIPLE_DIM,
)
from models.TaskA.residual_fusion_decoder import ResidualFusionHCRDecoder


def test_gate_slots_are_pu_pv_p11_rarity_support():
    assert GATE_SLOTS == (20, 24, 28, 31, 32)


def test_shapes_and_nonbinary_residual_zero():
    dec = ResidualFusionHCRDecoder(
        hidden_channels=32, legacy_mode="v0_8", use_triple=False
    )
    b = 4
    emb = torch.randn(20, 32)
    src = torch.tensor([0, 1, 2, 3])
    tgt = torch.tensor([4, 5, 6, 7])
    b2 = torch.randn(b, MOTIF_B2_DIM)
    v0 = torch.randn(b, MOTIF_V0_DIM)
    mask = torch.zeros(b, 3)
    mask[:2, 0] = 1.0

    logits = dec(emb, src, tgt, b2, v0, mask)
    assert logits.shape == (b,)
    assert torch.isfinite(logits).all()
    motif, alpha = dec.encode_motif(b2, v0, mask)
    assert motif.shape[1] == 24
    assert torch.allclose(alpha[:, 1:, :], torch.zeros_like(alpha[:, 1:, :]))


def test_disabled_legacy_matches_pure_b2_branch():
    torch.manual_seed(0)
    r2 = ResidualFusionHCRDecoder(32, legacy_mode="v0_8", use_triple=False)
    r2.eval()
    b = 3
    emb = torch.randn(10, 32)
    src = torch.arange(b)
    tgt = torch.arange(b) + 3
    b2 = torch.randn(b, MOTIF_B2_DIM)
    v0 = torch.randn(b, MOTIF_V0_DIM)
    mask = torch.ones(b, 3)
    with torch.no_grad():
        r2.legacy_encoder[-1].weight.fill_(0.25)
        r2.legacy_encoder[-1].bias.fill_(0.25)
        r2.gate[-1].bias.fill_(2.0)
    y_off = r2(emb, src, tgt, b2, v0, mask, force_disable_legacy=True)
    motif_off, alpha_off = r2.encode_motif(b2, v0, mask, force_disable_legacy=True)
    motif_on, alpha_on = r2.encode_motif(b2, v0, mask, force_disable_legacy=False)
    motif_b2, _ = r2.encode_motif(b2, None, None)
    assert torch.allclose(motif_off, motif_b2, atol=1e-6)
    assert torch.allclose(alpha_off, torch.zeros_like(alpha_off))
    assert float(alpha_on.mean()) > 0.5
    assert not torch.allclose(motif_on, motif_off, atol=1e-4)
    y_on = r2(emb, src, tgt, b2, v0, mask, force_disable_legacy=False)
    assert not torch.allclose(y_on, y_off, atol=1e-4)


def test_r4_disable_triple_matches_r3_head_width_path():
    torch.manual_seed(1)
    r4 = ResidualFusionHCRDecoder(32, legacy_mode="classical4", use_triple=True)
    r4.eval()
    b = 2
    emb = torch.randn(8, 32)
    src = torch.arange(b)
    tgt = torch.arange(b) + 2
    b2 = torch.randn(b, MOTIF_B2_DIM)
    c4 = torch.randn(b, MOTIF_CLASSICAL4_DIM)
    mask = torch.ones(b, 3)
    triple = torch.randn(b, TRIPLE_DIM)
    with torch.no_grad():
        r4.triple_encoder[-2].weight.add_(0.3)
    y_on = r4(emb, src, tgt, b2, c4, mask, triple, force_disable_triple=False)
    y_off = r4(emb, src, tgt, b2, c4, mask, triple, force_disable_triple=True)
    motif_off, _ = r4.encode_motif(
        b2, c4, mask, triple, force_disable_triple=True
    )
    assert motif_off.shape == (b, MOTIF_LATENT_WITH_TRIPLE)
    assert torch.allclose(motif_off[:, 24:], torch.zeros(b, 4), atol=1e-6)
    assert not torch.allclose(y_on, y_off, atol=1e-4)


def test_conditional_edge_gain_positive_and_signed_rd():
    rng = np.random.default_rng(0)
    n = 800
    z = rng.integers(0, 2, size=n).astype(float)
    a = rng.integers(0, 2, size=n).astype(float)
    # G depends on A beyond Z
    logits = -0.5 + 1.2 * a + 0.3 * z
    g = (rng.random(n) < 1 / (1 + np.exp(-logits))).astype(float)
    res = compute_conditional_edge_gain(a, z, g, n_train=n)
    assert res.supported
    assert res.ig >= 0.0
    assert 0.0 <= res.ig_norm <= 1.0 + 1e-6
    assert -1.0 <= res.rd_cond <= 1.0
    assert res.vector.shape == (6,)
    assert res.vector[5] == 1.0


def test_cmi_cache_orientation_matters():
    rng = np.random.default_rng(2)
    n = 600
    a = rng.integers(0, 2, size=n).astype(float)
    z = rng.integers(0, 2, size=n).astype(float)
    g = ((a + z + rng.integers(0, 2, size=n)) % 2).astype(float)
    fwd = compute_conditional_edge_gain(a, z, g, n_train=n)
    rev = compute_conditional_edge_gain(g, z, a, n_train=n)
    # Directed IG A→G|Z vs G→A|Z need not match.
    if fwd.supported and rev.supported:
        assert not np.allclose(fwd.vector[:4], rev.vector[:4])


def test_pair_latent_dim():
    dec = ResidualFusionHCRDecoder(32, legacy_mode="none", use_triple=False)
    b2 = torch.randn(2, MOTIF_B2_DIM)
    motif, _ = dec.encode_motif(b2, None, None)
    assert motif.reshape(2, 3, PAIR_LATENT_DIM).shape[-1] == PAIR_LATENT_DIM
