"""Wave 9 gate tests for MLPPairEncoder (K0 path). KAN tests land after implementation."""

from __future__ import annotations

import torch

from models.TaskA.pair_encoders import PairEncoderConfig, build_pair_encoder
from models.TaskA.wave7c_decoder import _mlp_pair_encoder


def test_mlp_matches_legacy_wave7c_stack_output_shape_and_finiteness():
    cfg = PairEncoderConfig(type="mlp", dropout=0.1)
    enc = build_pair_encoder(cfg)
    legacy = _mlp_pair_encoder(0.1)
    # Copy weights so numerical path is comparable after init alignment.
    with torch.no_grad():
        for p_new, p_old in zip(enc.parameters(), legacy.parameters()):
            p_new.copy_(p_old)
    x = torch.randn(16, 40)
    enc.eval()
    legacy.eval()
    y = enc(x)
    y_legacy = legacy(x)
    assert y.shape == (16, 8)
    assert torch.isfinite(y).all()
    assert torch.allclose(y, y_legacy, atol=1e-6)


def test_role_mask_exact_zero():
    enc = build_pair_encoder(PairEncoderConfig(type="mlp"))
    x = torch.randn(4, 40)
    lat = enc(x)
    mask = torch.tensor([1.0, 0.0, 1.0, 0.0]).unsqueeze(-1)
    out = lat * mask
    assert torch.allclose(out[1], torch.zeros(8))
    assert torch.allclose(out[3], torch.zeros(8))


def test_kan_builds():
    enc = build_pair_encoder(PairEncoderConfig(type="kan", dropout=0.1))
    y = enc(torch.randn(2, 40))
    assert y.shape == (2, 8)
