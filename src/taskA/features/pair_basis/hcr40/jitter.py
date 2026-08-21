"""Deterministic distributional jitter for count → (0,1) uniforms."""

from __future__ import annotations

import hashlib

import numpy as np


def deterministic_xi(
    patient_ids: np.ndarray | list,
    *,
    variable_name: str,
    seed: int,
) -> np.ndarray:
    """ξ ∈ (0,1) from patient_id + variable_name + seed (stable across calls)."""
    out = np.empty(len(patient_ids), dtype=np.float64)
    var = str(variable_name)
    seed_s = str(int(seed))
    for i, pid in enumerate(patient_ids):
        payload = f"{seed_s}|{pid}|{var}|jitter".encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        # Open interval (0,1)
        out[i] = (int(digest[:16], 16) + 0.5) / float(2**64)
    return np.clip(out, 1e-12, 1.0 - 1e-12)


def count_to_jittered_u(
    counts: np.ndarray,
    patient_ids: np.ndarray | list,
    *,
    variable_name: str,
    seed: int,
    categories: list[float],
    probabilities: np.ndarray,
    left_cdf: np.ndarray,
) -> np.ndarray:
    """Map capped count → u = F(c-) + ξ p(c) using train PMF tables."""
    x = np.asarray(counts, dtype=float)
    xi = deterministic_xi(patient_ids, variable_name=variable_name, seed=seed)
    cat_to_idx = {float(c): i for i, c in enumerate(categories)}
    u = np.full(len(x), np.nan, dtype=np.float64)
    for i, val in enumerate(x):
        if not np.isfinite(val):
            continue
        key = float(val)
        if key not in cat_to_idx:
            # Unknown / above-cap already applied; treat as last category if present.
            if categories:
                key = float(categories[-1])
            else:
                continue
        j = cat_to_idx[key]
        u[i] = float(left_cdf[j] + xi[i] * probabilities[j])
    return np.clip(u, 1e-6, 1.0 - 1e-6)
