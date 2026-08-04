"""Unit tests for Wave 9 KAN pair encoder (gates before training)."""

from __future__ import annotations

import copy

import torch

from models.TaskA.pair_encoders import PairEncoderConfig, build_pair_encoder
from models.TaskA.pair_encoders.kan_linear import KANLinear


def _kan_cfg(**kwargs) -> PairEncoderConfig:
    base = dict(
        type="kan",
        input_dim=40,
        hidden_dim=16,
        output_dim=8,
        spline_order=3,
        grid_size=5,
        grid_range=(-3.0, 3.0),
        grid_update=False,
        base_activation="silu",
        base_scale_init=1.0,
        spline_scale_init=0.1,
        use_bias=True,
        dropout=0.1,
        spline_l1=1.0e-5,
    )
    base.update(kwargs)
    return PairEncoderConfig(**base)


def test_kan_shape_finite_and_grad():
    enc = build_pair_encoder(_kan_cfg())
    x = torch.randn(32, 40, requires_grad=True)
    y = enc(x)
    assert y.shape == (32, 8)
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for p in enc.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_kan_determinism_and_checkpoint():
    torch.manual_seed(0)
    enc = build_pair_encoder(_kan_cfg())
    enc.eval()
    x = torch.randn(8, 40)
    with torch.no_grad():
        y1 = enc(x)
        y2 = enc(x)
    assert torch.allclose(y1, y2)
    state = copy.deepcopy(enc.state_dict())
    enc2 = build_pair_encoder(_kan_cfg())
    enc2.load_state_dict(state)
    enc2.eval()
    with torch.no_grad():
        y3 = enc2(x)
    assert torch.allclose(y1, y3, atol=1e-6)


def test_kan_zero_and_constant_input_stable():
    enc = build_pair_encoder(_kan_cfg())
    enc.eval()
    with torch.no_grad():
        z = enc(torch.zeros(4, 40))
        c = enc(torch.full((4, 40), 2.5))
    assert torch.isfinite(z).all()
    assert torch.isfinite(c).all()


def test_role_mask_exact_zero_after_encode():
    enc = build_pair_encoder(_kan_cfg())
    x = torch.randn(3, 40)
    lat = enc(x)
    mask = torch.tensor([1.0, 0.0, 1.0]).unsqueeze(-1)
    out = lat * mask
    assert torch.allclose(out[1], torch.zeros(8), atol=0.0)


def test_grid_update_false_is_noop_on_valid_like_call():
    layer = KANLinear(40, 16, grid_size=5, spline_order=3, grid_range=(-3, 3), grid_update=False)
    grid_before = layer.grid.clone()
    x = torch.randn(64, 40)
    layer.update_grid_from_data(x)
    assert torch.equal(layer.grid, grid_before)


def test_wave7c_unshared_kan_forward():
    from models.TaskA.wave7c_decoder import Wave7CDecoder
    from hcr.wave7.panel_b_residual.constants import MOTIF_B2_DIM

    dec = Wave7CDecoder(
        32,
        arch="unshared_mlp",
        pair_encoder_cfg=_kan_cfg(),
    )
    emb = torch.randn(10, 32)
    y = dec(emb, torch.arange(2), torch.arange(2) + 2, torch.randn(2, MOTIF_B2_DIM))
    assert y.shape == (2,)
    assert torch.isfinite(y).all()
    reg = dec.pair_encoder_regularization_loss()
    assert torch.isfinite(reg)
    assert float(reg) > 0.0
