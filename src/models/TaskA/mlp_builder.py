"""Configurable MLP builders for Task A decoders."""

from __future__ import annotations

from torch import nn

from models.TaskA.activations import build_activation


def build_mlp(
    *,
    input_dim: int,
    hidden_dims: list[int],
    activation: str,
    dropout: float,
) -> nn.Sequential:
    if not hidden_dims:
        return nn.Sequential(nn.Linear(input_dim, 1))

    layers: list[nn.Module] = []
    current_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend(
            [
                nn.Linear(current_dim, hidden_dim),
                build_activation(activation),
                nn.Dropout(dropout),
            ]
        )
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, 1))
    return nn.Sequential(*layers)
