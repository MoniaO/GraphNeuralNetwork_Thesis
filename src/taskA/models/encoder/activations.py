"""Encoder activations. FINAL HGT: leaky_relu (slope 0.1). Do not change for 14.08."""

from __future__ import annotations

from torch import nn


def build_activation(
    name: str,
    *,
    leaky_relu_slope: float = 0.1,
) -> nn.Module:
    normalized = name.strip().lower()

    if normalized == "relu":
        return nn.ReLU()
    if normalized == "leaky_relu":
        return nn.LeakyReLU(negative_slope=leaky_relu_slope)
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "silu":
        return nn.SiLU()
    if normalized == "mish":
        return nn.Mish()
    if normalized == "identity":
        return nn.Identity()

    raise ValueError(f"Unsupported activation: {name}")
