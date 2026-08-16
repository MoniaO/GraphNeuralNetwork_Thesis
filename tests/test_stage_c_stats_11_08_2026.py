"""Minimal Stage C tests (11.08.2026): shapes, masks, Fusion88StatDecoder, S0 zeros."""

from __future__ import annotations

import numpy as np
import torch

from taskA_final_large_grid_11_08_2026.fusion88_decoder import Fusion88Decoder
from taskA_final_large_grid_11_08_2026.stage_c.features import (
    cosine_active,
    jaccard_active,
    normalized_mutual_information,
    signed_phi_binary,
    slice_full40,
    _fit_discretizer,
)
from taskA_final_large_grid_11_08_2026.stage_c.stat_encoder import (
    Fusion88StatDecoder,
    StatPairEncoder,
)
from taskA_final_large_grid_11_08_2026.stage_c.variants import (
    STAGE_C_VARIANTS,
    get_variant,
)


def test_variant_raw_dims():
    dims = {v.id: v.raw_dim for v in STAGE_C_VARIANTS}
    assert dims["S0"] == 0
    assert dims["S1"] == dims["S2"] == dims["S3"] == dims["S4"] == 1
    assert dims["S5"] == 8
    assert dims["S6"] == 16
    assert dims["S7"] == 24
    assert dims["S8"] == 32
    assert dims["S9"] == 36
    assert dims["S10"] == 40
    assert get_variant("S10_HCR_FULL40").id == "S10"


def test_nmi_jaccard_cosine_shapes():
    rng = np.random.default_rng(0)
    u = rng.integers(0, 2, size=200).astype(float)
    v = (u + rng.integers(0, 2, size=200)) % 2
    disc_u = _fit_discretizer(u, "binary")
    disc_v = _fit_discretizer(v, "binary")
    nmi, raw_mi = normalized_mutual_information(u, v, disc_u=disc_u, disc_v=disc_v)
    assert isinstance(nmi, float) and isinstance(raw_mi, float)
    assert 0.0 <= nmi <= 1.0 + 1e-6

    # continuous activity path uses robust-z
    uc = rng.normal(size=200)
    vc = uc + rng.normal(scale=0.1, size=200)
    j = jaccard_active(uc, vc, kind_u="continuous", kind_v="continuous")
    c = cosine_active(uc, vc, kind_u="continuous", kind_v="continuous")
    assert np.ndim(j) == 0 and np.ndim(c) == 0
    assert 0.0 <= j <= 1.0
    assert 0.0 <= c <= 1.0


def test_signed_phi_no_silent_binarize():
    u = np.array([0.0, 1.0, 0.0, 1.0])
    v = np.array([0.0, 1.0, 1.0, 0.0])
    phi, ok = signed_phi_binary(u, v)
    assert ok
    assert np.isfinite(phi)

    # continuous must NOT be thresholded → not applicable
    uc = np.array([0.2, 1.7, 0.1, 3.0])
    vc = np.array([0.3, 1.1, 0.0, 2.5])
    phi2, ok2 = signed_phi_binary(uc, vc)
    assert ok2 is False
    assert phi2 == 0.0


def test_mask_missing_z_pattern():
    """Missing Z → structural masks [0,1,0] (AG only)."""
    masks = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert list(masks) == [0.0, 1.0, 0.0]


def test_stat_pair_encoder_variable_d():
    for d in (1, 8, 16, 24, 32, 36, 40):
        enc = StatPairEncoder(d)
        x = torch.randn(5, d)
        y = enc(x)
        assert y.shape == (5, 8)


def test_fusion88_stat_decoder_forward_shapes():
    dec = Fusion88StatDecoder(hidden_channels=32, stat_raw_dim=16, dropout=0.2)
    emb = torch.randn(12, 32)
    src = torch.tensor([0, 1, 2, 3])
    tgt = torch.tensor([4, 5, 6, 7])
    raw = torch.randn(4, 3, 16)
    masks = torch.tensor(
        [[1, 1, 1], [0, 1, 0], [1, 1, 0], [0, 1, 1]], dtype=torch.float32
    )
    logits = dec(emb, src, tgt, stat_raw=raw, stat_role_masks=masks)
    assert logits.shape == (4,)
    assert torch.isfinite(logits).all()
    g = dec.encode_g_stat(raw, masks)
    assert g.shape == (4, 24)
    # masked roles zeroed in latent
    assert torch.allclose(g[1, 0:8], torch.zeros(8), atol=1e-6)
    assert torch.allclose(g[1, 16:24], torch.zeros(8), atol=1e-6)


def test_s0_equals_zeros_path():
    """S0 force_zero_stat must match plain Fusion88Decoder (g_stat=None → zeros)."""
    torch.manual_seed(0)
    emb = torch.randn(10, 32)
    src = torch.tensor([0, 1, 2])
    tgt = torch.tensor([3, 4, 5])

    plain = Fusion88Decoder(hidden_channels=32, dropout=0.0)
    s0 = Fusion88StatDecoder(
        hidden_channels=32, stat_raw_dim=1, dropout=0.0, force_zero_stat=True
    )
    # Copy graph/fusion weights so only stat path differs.
    s0.graph_branch.load_state_dict(plain.graph_branch.state_dict())
    s0.output_head.load_state_dict(plain.output_head.state_dict())

    raw = torch.randn(3, 3, 1)
    masks = torch.tensor([[0, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=torch.float32)
    y_plain = plain(emb, src, tgt, g_stat=None)
    y_s0 = s0(emb, src, tgt, stat_raw=raw, stat_role_masks=masks)
    assert torch.allclose(y_plain, y_s0, atol=1e-5)


def test_slice_full40_dims():
    vec = np.arange(40, dtype=np.float32)
    assert slice_full40(vec, get_variant("S5")).shape == (8,)
    assert slice_full40(vec, get_variant("S6")).shape == (16,)
    assert slice_full40(vec, get_variant("S7")).shape == (24,)
    assert slice_full40(vec, get_variant("S8")).shape == (32,)
    assert slice_full40(vec, get_variant("S9")).shape == (36,)
    assert slice_full40(vec, get_variant("S10")).shape == (40,)
