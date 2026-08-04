"""Map variable-size HCR coefficient matrices into the frozen 8-d pair block."""

from __future__ import annotations

import numpy as np


PAIR_DIM = 8


def pair_kinds(kind_u: str, kind_v: str) -> str:
    ku, kv = kind_u.lower(), kind_v.lower()
    discrete = {"binary", "count", "discrete"}
    if ku == "binary" and kv == "binary":
        return "binary_binary"
    if ku == "binary" and kv == "count":
        return "binary_count"
    if ku == "count" and kv == "binary":
        return "count_binary"
    if ku in discrete and kv == "continuous":
        return "discrete_continuous"
    if ku == "continuous" and kv in discrete:
        return "continuous_discrete"
    if ku == "count" and kv == "count":
        return "count_count"
    if ku == "continuous" and kv == "continuous":
        return "continuous_continuous"
    # fallback treat unknown discrete-ish as count-like first-4 flatten
    return "count_count"


def select_explicit_coefficients(matrix: np.ndarray, mapping: str) -> np.ndarray:
    """Return the four explicit coefficients c1..c4 for the frozen block."""
    a = np.asarray(matrix, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    c = np.zeros(4, dtype=float)
    if a.size == 0:
        return c

    if mapping == "binary_binary":
        c[0] = float(a[0, 0])
    elif mapping == "binary_count":
        row = a[0, :]
        n = min(4, row.size)
        c[:n] = row[:n]
    elif mapping == "count_binary":
        col = a[:, 0]
        n = min(4, col.size)
        c[:n] = col[:n]
    else:
        # [a11, a12, a21, a22]
        if a.shape[0] >= 1 and a.shape[1] >= 1:
            c[0] = float(a[0, 0])
        if a.shape[0] >= 1 and a.shape[1] >= 2:
            c[1] = float(a[0, 1])
        if a.shape[0] >= 2 and a.shape[1] >= 1:
            c[2] = float(a[1, 0])
        if a.shape[0] >= 2 and a.shape[1] >= 2:
            c[3] = float(a[1, 1])
    return c


def energy_all_coefficients(matrix: np.ndarray) -> float:
    return float(np.sum(np.asarray(matrix, dtype=float) ** 2))


def pack_pair_vector(
    matrix: np.ndarray,
    *,
    kind_u: str,
    kind_v: str,
    support: float,
    bootstrap_sd: float,
    supported: bool,
    binary_energy_is_a11_sq: bool = False,
) -> np.ndarray:
    """[c1,c2,c3,c4,energy,support,bootstrap_sd,supported_mask]."""
    mapping = pair_kinds(kind_u, kind_v)
    c = select_explicit_coefficients(matrix, mapping)
    if binary_energy_is_a11_sq and mapping == "binary_binary":
        energy = float(c[0] ** 2)
    else:
        energy = energy_all_coefficients(matrix)
    out = np.zeros(PAIR_DIM, dtype=np.float32)
    out[:4] = c.astype(np.float32)
    out[4] = np.float32(energy)
    out[5] = np.float32(support)
    out[6] = np.float32(bootstrap_sd)
    out[7] = np.float32(1.0 if supported else 0.0)
    return out


def coefficient_stats(matrix: np.ndarray, mapping: str) -> dict:
    a = np.asarray(matrix, dtype=float)
    flat = a.ravel()
    explicit = select_explicit_coefficients(a, mapping)
    n_explicit = int(np.count_nonzero(np.abs(explicit) > 0) or min(4, flat.size))
    return {
        "full_coefficient_shape": list(a.shape),
        "number_of_coefficients": int(flat.size),
        "number_retained_explicitly": min(4, int(flat.size)),
        "number_summarized_only_in_energy": max(0, int(flat.size) - min(4, int(flat.size))),
        "mapping": mapping,
    }
