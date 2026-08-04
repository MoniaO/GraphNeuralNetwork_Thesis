"""Unit tests for Wave 7C architecture/context decoder controls."""

from __future__ import annotations

import torch

from hcr.wave7.panel_b_residual.constants import MOTIF_B2_DIM, PAIR_B2_DIM
from models.TaskA.wave7c_decoder import Wave7CDecoder


def test_zero_input_through_shared_mlp_is_nonzero():
    dec = Wave7CDecoder(32, arch="shared_mlp", ablation="none")
    dec.eval()
    with torch.no_grad():
        z = dec.pair_encoder(torch.zeros(1, PAIR_B2_DIM))
    assert z.shape == (1, 8)
    assert float(z.norm()) > 1e-6


def test_c6_forces_exact_zero_latent_for_masked_roles():
    torch.manual_seed(0)
    dec = Wave7CDecoder(32, arch="shared_mlp", ablation="explicit_role_masks")
    dec.eval()
    b = 2
    emb = torch.randn(10, 32)
    src = torch.arange(b)
    tgt = torch.arange(b) + 2
    # Non-zero raw AZ/ZG blocks, but masks say missing.
    b2 = torch.randn(b, MOTIF_B2_DIM)
    masks = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    with torch.no_grad():
        role = dec.encode_pairs(b2)
        motif, m = dec.apply_ablation(b2, role, masks)
    assert m is not None
    lat = motif[:, :24].reshape(b, 3, 8)
    assert torch.allclose(lat[:, 0, :], torch.zeros(b, 8))
    assert torch.allclose(lat[:, 2, :], torch.zeros(b, 8))
    assert not torch.allclose(lat[:, 1, :], torch.zeros(b, 8))
    # Head width includes 3 masks → 91D concat inside forward.
    y = dec(emb, src, tgt, b2, masks)
    assert y.shape == (b,)


def test_c3_keeps_only_support_mask_type_slots():
    dec = Wave7CDecoder(32, arch="shared_mlp", ablation="mask_type_support_only")
    b2 = torch.randn(3, MOTIF_B2_DIM)
    kept = dec.mask_type_support_only(b2)
    blocks = kept.reshape(3, 3, 40)
    assert torch.allclose(blocks[:, :, :32], torch.zeros_like(blocks[:, :, :32]))
    assert torch.allclose(blocks[:, :, 32:40], b2.reshape(3, 3, 40)[:, :, 32:40])


def test_arch_resolvers_and_forward_shapes():
    from models.TaskA.wave7c_decoder import resolve_wave7c_ablation, resolve_wave7c_arch

    assert resolve_wave7c_arch("W7C_A1_UNSHARED_ROLE_ENCODERS") == "unshared_mlp"
    assert resolve_wave7c_arch("W7C_A2_GLOBAL_MOTIF_ENCODER") == "global_motif"
    assert resolve_wave7c_arch("W7C_A3_SHARED_LINEAR_PAIR_ENCODER") == "shared_linear"
    assert resolve_wave7c_ablation("W7C_C1_AG_ONLY") == "ag_only"
    assert resolve_wave7c_ablation("W7C_C6_EXPLICIT_ROLE_MASKS") == "explicit_role_masks"

    for arch in ("shared_mlp", "unshared_mlp", "global_motif", "shared_linear"):
        dec = Wave7CDecoder(32, arch=arch, ablation="none")
        emb = torch.randn(8, 32)
        y = dec(emb, torch.arange(2), torch.arange(2) + 2, torch.randn(2, MOTIF_B2_DIM))
        assert y.shape == (2,)


def test_t1_triple_masked_zero_and_head_width():
    from hcr.wave7.panel_b_residual.constants import TRIPLE_DIM

    torch.manual_seed(0)
    dec = Wave7CDecoder(32, arch="unshared_mlp", ablation="none", use_triple=True)
    dec.eval()
    b = 3
    emb = torch.randn(10, 32)
    b2 = torch.randn(b, MOTIF_B2_DIM)
    triple = torch.randn(b, TRIPLE_DIM)
    triple[0, 5] = 0.0  # unsupported / missing Z
    triple[1:, 5] = 1.0
    with torch.no_grad():
        # Force nonzero encoder output then verify mask zeros row 0.
        for p in dec.triple_encoder.parameters():
            p.add_(0.1)
        y = dec(emb, torch.arange(b), torch.arange(b) + 3, b2, None, triple)
        t_lat = dec.triple_encoder(triple) * triple[:, 5:6]
    assert y.shape == (b,)
    assert torch.allclose(t_lat[0], torch.zeros(4), atol=1e-6)
    assert not torch.allclose(t_lat[1], torch.zeros(4), atol=1e-6)
