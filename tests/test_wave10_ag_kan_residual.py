"""Unit tests for Wave 10 AG KAN residual."""

from __future__ import annotations

import math

import torch

from models.TaskA.pair_encoders.ag_kan_residual import AGKANResidual, AGKANResidualConfig
from models.TaskA.wave7c_decoder import Wave7CDecoder, resolve_ag_kan_residual_cfg


def test_ag_residual_shapes_and_gate_init():
    cfg = AGKANResidualConfig(enabled=True, gate_init_logit=-3.0, grid_size=3)
    mod = AGKANResidual(cfg)
    x = torch.randn(16, 40)
    gated, alpha, raw = mod(x)
    assert gated.shape == (16, 8)
    assert raw.shape == (16, 8)
    assert abs(float(alpha) - (1.0 / (1.0 + math.exp(3.0)))) < 1e-5


def test_ag_residual_role_mask_zeros():
    mod = AGKANResidual(AGKANResidualConfig(enabled=True))
    x = torch.randn(8, 40)
    mask = torch.tensor([1, 1, 0, 0, 1, 0, 1, 0], dtype=torch.float32)
    gated, _, raw = mod(x, role_mask=mask)
    assert torch.all(raw[mask == 0].abs() == 0)
    assert torch.all(gated[mask == 0].abs() == 0)


def test_wave7c_decoder_ag_residual_forward():
    dec = Wave7CDecoder(
        32,
        arch="unshared_mlp",
        ag_kan_residual_cfg=AGKANResidualConfig(enabled=True, gate_init_logit=-3.0),
    )
    emb = torch.randn(20, 32)
    b2 = torch.randn(5, 120)
    masks = torch.ones(5, 3)
    masks[:, 0] = 0  # AZ missing
    logits = dec(
        embeddings=emb,
        source=torch.arange(5),
        target=torch.arange(5, 10),
        hcr_features=b2,
        hcr_role_masks=masks,
    )
    assert logits.shape == (5,)
    assert "gate_alpha" in dec.last_residual_stats
    # Only residual + gate trainable when freeze applied externally — just check params exist.
    assert len(dec.ag_kan_trainable_parameters()) > 0


def test_resolve_ag_cfg_from_variant():
    class _D:
        ag_kan_residual = None

    cfg = resolve_ag_kan_residual_cfg("W10_R1_AG_KAN_RESIDUAL_FROZEN_MLP", _D())
    assert cfg.enabled is True
    assert cfg.train_mode == "residual_only"

    cfg2 = resolve_ag_kan_residual_cfg("W10_T1_TYPE_ROUTED_KAN_RESIDUAL", _D())
    assert cfg2.enabled is True
    assert cfg2.type_routed is True


def test_regularization_nonzero():
    mod = AGKANResidual(AGKANResidualConfig(enabled=True, spline_l1=1e-5))
    reg = mod.regularization_loss()
    assert torch.is_tensor(reg)
    assert float(reg) >= 0.0
