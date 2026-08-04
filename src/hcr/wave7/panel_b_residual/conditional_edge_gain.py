"""Triplewise conditional edge gain I(A;G|Z) for binary–binary–binary motifs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-12
SMOOTHING = 0.5
MIN_COMPLETE = 100
MIN_AZ_STRATUM = 10
TRIPLE_DIM = 6


def binary_entropy(p: float) -> float:
    p = float(np.clip(p, EPS, 1.0 - EPS))
    return float(-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)))


@dataclass(frozen=True)
class ConditionalEdgeGainResult:
    vector: np.ndarray  # 6D
    n_raw: np.ndarray  # shape (2,2,2) raw counts
    n_complete: int
    h_g_given_z: float
    h_g_given_az: float
    ig: float
    ig_norm: float
    rd_cond: float
    rd_z0: float
    rd_z1: float
    interaction: float
    support: float
    mask: float
    supported: bool
    reason: str


def _zeros(reason: str, n_train: int, n_complete: int = 0) -> ConditionalEdgeGainResult:
    return ConditionalEdgeGainResult(
        vector=np.zeros(TRIPLE_DIM, dtype=np.float32),
        n_raw=np.zeros((2, 2, 2), dtype=np.int64),
        n_complete=int(n_complete),
        h_g_given_z=0.0,
        h_g_given_az=0.0,
        ig=0.0,
        ig_norm=0.0,
        rd_cond=0.0,
        rd_z0=0.0,
        rd_z1=0.0,
        interaction=0.0,
        support=float(n_complete / max(n_train, 1)),
        mask=0.0,
        supported=False,
        reason=reason,
    )


def compute_conditional_edge_gain(
    a: np.ndarray,
    z: np.ndarray,
    g: np.ndarray,
    *,
    n_train: int,
) -> ConditionalEdgeGainResult:
    """IG = H(G|Z) - H(G|A,Z) for directed candidate A→G with context Z.

    Uses patient_train complete cases only (caller must pass train rows).
    """
    aa = np.asarray(a, dtype=float)
    zz = np.asarray(z, dtype=float)
    gg = np.asarray(g, dtype=float)
    complete = np.isfinite(aa) & np.isfinite(zz) & np.isfinite(gg)
    n_complete = int(complete.sum())
    if n_complete < MIN_COMPLETE:
        return _zeros("insufficient_complete", n_train, n_complete)

    a_i = np.clip(np.round(aa[complete]), 0, 1).astype(np.int64)
    z_i = np.clip(np.round(zz[complete]), 0, 1).astype(np.int64)
    g_i = np.clip(np.round(gg[complete]), 0, 1).astype(np.int64)

    # Both levels for each variable.
    if len(np.unique(a_i)) < 2 or len(np.unique(z_i)) < 2 or len(np.unique(g_i)) < 2:
        return _zeros("missing_level", n_train, n_complete)

    n_raw = np.zeros((2, 2, 2), dtype=np.int64)
    for ai, zi, gi in zip(a_i, z_i, g_i):
        n_raw[int(ai), int(zi), int(gi)] += 1

    # All four (A,Z) strata present with enough raw mass.
    for ai in (0, 1):
        for zi in (0, 1):
            n_az = int(n_raw[ai, zi, :].sum())
            if n_az < MIN_AZ_STRATUM:
                return _zeros("az_stratum_sparse_or_missing", n_train, n_complete)

    n_smooth = n_raw.astype(np.float64) + SMOOTHING
    p = n_smooth / n_smooth.sum()

    # H(G|Z)
    h_g_z = 0.0
    rd_z = np.zeros(2, dtype=np.float64)
    p_z = np.zeros(2, dtype=np.float64)
    for zi in (0, 1):
        p_z[zi] = float(p[:, zi, :].sum())
        p_g1_z = float(p[:, zi, 1].sum() / max(p_z[zi], EPS))
        h_g_z += p_z[zi] * binary_entropy(p_g1_z)
        # RD_z = P(G=1|A=1,z) - P(G=1|A=0,z)
        dens1 = float(p[1, zi, :].sum())
        dens0 = float(p[0, zi, :].sum())
        p_g1_a1 = float(p[1, zi, 1] / max(dens1, EPS))
        p_g1_a0 = float(p[0, zi, 1] / max(dens0, EPS))
        rd_z[zi] = p_g1_a1 - p_g1_a0

    # H(G|A,Z)
    h_g_az = 0.0
    for ai in (0, 1):
        for zi in (0, 1):
            p_az = float(p[ai, zi, :].sum())
            p_g1_az = float(p[ai, zi, 1] / max(p_az, EPS))
            h_g_az += p_az * binary_entropy(p_g1_az)

    ig = max(float(h_g_z - h_g_az), 0.0)
    ig_norm = float(ig / (h_g_z + EPS))
    ig_norm = float(np.clip(ig_norm, 0.0, 1.0))
    rd_cond = float(p_z[0] * rd_z[0] + p_z[1] * rd_z[1])
    rd_cond = float(np.clip(rd_cond, -1.0, 1.0))
    interaction = float(rd_z[1] - rd_z[0])
    interaction = float(np.clip(interaction, -2.0, 2.0))
    support = float(n_complete / max(n_train, 1))
    mask = 1.0
    vec = np.array(
        [ig, ig_norm, rd_cond, interaction, support, mask], dtype=np.float32
    )
    assert ig >= -1e-10
    assert 0.0 - 1e-8 <= ig_norm <= 1.0 + 1e-6
    assert -1.0 - 1e-8 <= rd_cond <= 1.0 + 1e-8
    assert -2.0 - 1e-8 <= interaction <= 2.0 + 1e-8
    return ConditionalEdgeGainResult(
        vector=vec,
        n_raw=n_raw,
        n_complete=n_complete,
        h_g_given_z=float(h_g_z),
        h_g_given_az=float(h_g_az),
        ig=ig,
        ig_norm=ig_norm,
        rd_cond=rd_cond,
        rd_z0=float(rd_z[0]),
        rd_z1=float(rd_z[1]),
        interaction=interaction,
        support=support,
        mask=mask,
        supported=True,
        reason="ok",
    )
