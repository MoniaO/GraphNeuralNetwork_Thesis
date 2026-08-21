"""Tests for FINAL 14.08.2026 — StatKAN twin + S10 shapes."""

from __future__ import annotations

import torch

from taskA.models.decoder.pair_encoder import (
    Fusion88StatDecoder,
    StatKANPairEncoder,
    StatPairEncoder,
)


def test_stat_kan_shapes_s10():
    enc = StatKANPairEncoder(40)
    x = torch.randn(16, 40)
    y = enc(x)
    assert y.shape == (16, 8)
    reg = enc.regularization_loss()
    assert torch.is_tensor(reg) and reg.ndim == 0


def test_fusion88_stat_kan_forward():
    dec = Fusion88StatDecoder(
        hidden_channels=32,
        stat_raw_dim=40,
        stat_pair_encoder="kan_shallow",
    )
    emb = torch.randn(50, 32)
    src = torch.randint(0, 50, (12,))
    tgt = torch.randint(0, 50, (12,))
    raw = torch.randn(12, 3, 40)
    masks = torch.tensor([[1.0, 1.0, 0.0]] * 12)
    logits = dec(emb, src, tgt, stat_raw=raw, stat_role_masks=masks)
    assert logits.shape == (12,)
    reg = dec.pair_encoder_regularization_loss()
    assert float(reg.detach()) >= 0.0


def test_fusion88_stat_mlp_reg_zero():
    dec = Fusion88StatDecoder(
        hidden_channels=32,
        stat_raw_dim=40,
        stat_pair_encoder="mlp",
    )
    assert float(dec.pair_encoder_regularization_loss().detach()) == 0.0
    assert isinstance(dec.role_encoders[0], StatPairEncoder)
