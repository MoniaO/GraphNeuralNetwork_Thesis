"""Smoke tests for Final Large Grid 11.08.2026 Fusion88 decoder."""

from __future__ import annotations

import torch

from taskA.models.decoder.fusion88 import (
    FUSION_DIM,
    Fusion88Decoder,
)
from taskA.experiments.stage_a_backbone.grid import (
    CANDIDATE_SEED,
    count_shared_jobs,
    iter_shared_grid,
)


def test_fusion88_zeros_stat_dim():
    dec = Fusion88Decoder(hidden_channels=32, dropout=0.2)
    emb = torch.randn(10, 32)
    src = torch.tensor([0, 1, 2])
    tgt = torch.tensor([3, 4, 5])
    logits = dec(emb, src, tgt, g_stat=None)
    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()


def test_fusion88_fusion_dim_88():
    dec = Fusion88Decoder(hidden_channels=64, dropout=0.2)
    assert dec.fusion_dim == FUSION_DIM == 88
    emb = torch.randn(20, 64)
    src = torch.arange(4)
    tgt = torch.arange(4, 8)
    g_stat = torch.zeros(4, 24)
    logits = dec(emb, src, tgt, g_stat=g_stat)
    assert logits.shape == (4,)


def test_candidate_seed_constant():
    assert CANDIDATE_SEED == 20260722


def test_shared_grid_nonempty_and_gat_divisible():
    configs = list(iter_shared_grid())
    assert len(configs) > 0
    for c in configs:
        if c.backbone == "hetero_gatv2":
            assert c.heads is not None
            assert c.hidden_dim % int(c.heads) == 0
    assert count_shared_jobs(3) == len(configs) * 3
