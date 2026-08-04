from __future__ import annotations

import numpy as np
import pandas as pd

from hcr.binary_features import binary_pair_features, result_to_vector
from hcr.pair_encoder import HCRPairEncoder, HCRPairEncoderConfig
from hcr.variable_spec import VariableSpec, VariableType


def test_hcr_features_depend_only_on_train_partition():
    specs = {
        "x": VariableSpec("x", VariableType.BINARY, "x"),
        "y": VariableSpec("y", VariableType.BINARY, "y"),
    }
    train = pd.DataFrame(
        {
            "x": [0, 0, 1, 1],
            "y": [0, 0, 1, 1],
            "split": ["train"] * 4,
        }
    )
    # Validation has the opposite association — must not affect fitted features.
    valid = pd.DataFrame(
        {
            "x": [0, 0, 1, 1],
            "y": [1, 1, 0, 0],
            "split": ["validation"] * 4,
        }
    )
    all_df = pd.concat([train, valid], ignore_index=True)

    enc_train = HCRPairEncoder(specs, HCRPairEncoderConfig(output_dim=8))
    enc_train.fit(train[["x", "y"]], [("x", "y")])
    v_train = enc_train.transform([("x", "y")]).numpy()[0]

    # Wrong methodology: fit on all patients (leakage).
    enc_leaky = HCRPairEncoder(specs, HCRPairEncoderConfig(output_dim=8))
    enc_leaky.fit(all_df[["x", "y"]], [("x", "y")])
    v_leaky = enc_leaky.transform([("x", "y")]).numpy()[0]

    assert not np.allclose(v_train, v_leaky)

    # Correct: fitting train equals computing binary features on train alone.
    direct = result_to_vector(binary_pair_features(train["x"].to_numpy(), train["y"].to_numpy()))
    assert np.allclose(v_train, direct, atol=1e-6)
