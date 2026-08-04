from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from hcr.pair_encoder import HCRPairEncoder, HCRPairEncoderConfig
from hcr.variable_spec import VariableSpec, VariableType


SPECS = {
    "a": VariableSpec("a", VariableType.BINARY, "a"),
    "b": VariableSpec("b", VariableType.BINARY, "b"),
    "c": VariableSpec("c", VariableType.CONTINUOUS, "c"),
}


def test_fit_transform_binary_pairs():
    df = pd.DataFrame(
        {
            "a": [0, 0, 1, 1, 1],
            "b": [0, 1, 0, 1, 1],
            "c": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    enc = HCRPairEncoder(
        SPECS,
        HCRPairEncoderConfig(output_dim=8, unsupported_pair_mode="zeros"),
    )
    enc.fit(df, [("a", "b"), ("a", "c")])
    out = enc.transform([("a", "b"), ("a", "c"), ("missing", "pair")])
    assert out.shape == (3, 8)
    assert torch.isfinite(out).all()
    # unsupported / unknown → zeros
    assert torch.allclose(out[1], torch.zeros(8))
    assert torch.allclose(out[2], torch.zeros(8))
    assert enc.n_binary_pairs == 1
    assert enc.n_unsupported_pairs == 1


def test_transform_requires_fit():
    enc = HCRPairEncoder(SPECS, HCRPairEncoderConfig())
    try:
        enc.transform([("a", "b")])
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "fitted" in str(exc).lower()
