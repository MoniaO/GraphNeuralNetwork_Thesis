"""V1B matched-aux packing checks (does not alter V1)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hcr.wave7.encoder import Wave7HCRConfig, Wave7PairEncoder


def test_v1b_uses_p11_and_rarity_not_completeness():
    rng = np.random.default_rng(0)
    n = 1000
    u = np.zeros(n)
    v = np.zeros(n)
    u[:100] = 1.0
    v[50:150] = 1.0  # n11 = 50
    df = pd.DataFrame({"a": u, "b": v})
    cfg = Wave7HCRConfig(
        variant="W7_V1B_GHCR_BINARY_MATCHED_AUX",
        bootstrap_repeats=10,
        v1_smoothing=0.0,
        min_complete=50,
        min_level_count=5,
    )
    enc = Wave7PairEncoder(cfg)
    enc.scenario = "clean"
    enc._kind = lambda name: "binary"  # type: ignore[method-assign]
    enc._column = lambda name: name  # type: ignore[method-assign]
    enc.fit(df, ["a", "b"])
    vec, meta = enc.transform_pair("a", "b")
    assert meta["supported"] is True
    assert meta["path"] == "ghcr_v1b_matched_aux"
    assert meta["bootstrap_in_vector"] is False
    n11 = 50
    assert np.isclose(vec[5], n11 / n)
    assert np.isclose(vec[6], 1.0 / np.sqrt(n11 + 1.0))
    # Must NOT be completeness≈1 stuffed into slot 5 as the main aux signal identity.
    assert not np.isclose(vec[5], 1.0)
    assert np.isclose(vec[4], float(vec[0]) ** 2, atol=1e-5)
