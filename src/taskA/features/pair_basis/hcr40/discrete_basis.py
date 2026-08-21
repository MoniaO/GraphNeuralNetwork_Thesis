"""One-hot → weighted Gram–Schmidt orthonormal contrasts for binary/count."""

from __future__ import annotations

import numpy as np


def fit_discrete_orthonormal_basis(
    values: np.ndarray,
    *,
    category_order: list[float] | None = None,
    smoothing: float = 0.5,
    tolerance: float = 1e-10,
    max_categories: int | None = None,
) -> tuple[list[float], np.ndarray, dict]:
    """Build K-1 orthonormal contrasts from full one-hot.

    Inner product: <f,g> = sum_k p_k f(k) g(k).

    Uses empirical p_k = n_k / N when every observed category has n_k > 0.
    Smoothing is applied only if a declared category has zero counts.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if category_order is None:
        categories = sorted(float(v) for v in np.unique(x).tolist())
    else:
        categories = [float(v) for v in category_order]

    k = len(categories)
    meta = {
        "K": k,
        "active_basis_dim": max(k - 1, 0),
        "categories": categories,
        "min_category_count": 0,
        "used_smoothing": False,
    }

    if max_categories is not None and k > int(max_categories):
        raise ValueError(
            f"Discrete variable has {k} train categories {categories}; "
            f"HCR40 one-hot limit is {max_categories}. "
            f"Do not silently convert to ranks — raise max_categories or pool the tail."
        )

    if k < 2:
        return categories, np.zeros((k, 0), dtype=float), meta

    category_to_index = {category: index for index, category in enumerate(categories)}
    counts = np.zeros(k, dtype=float)
    for value in x:
        counts[category_to_index[float(value)]] += 1.0

    meta["min_category_count"] = int(counts.min()) if counts.size else 0
    meta["counts"] = counts.astype(int).tolist()

    if np.any(counts <= 0):
        probabilities = (counts + float(smoothing)) / (counts.sum() + float(smoothing) * k)
        meta["used_smoothing"] = True
    else:
        probabilities = counts / counts.sum()

    raw = np.eye(k, dtype=float)
    constant = np.ones(k, dtype=float)
    constant /= np.sqrt(np.sum(probabilities * constant**2))
    basis = [constant]

    for column in range(k):
        candidate = raw[:, column].copy()
        for previous in basis:
            projection = np.sum(probabilities * candidate * previous)
            candidate -= projection * previous
        norm = np.sqrt(np.sum(probabilities * candidate**2))
        if norm > tolerance:
            basis.append(candidate / norm)
        if len(basis) == k:
            break

    contrasts = np.column_stack(basis[1:]) if len(basis) > 1 else np.zeros((k, 0), dtype=float)
    if contrasts.shape != (k, k - 1):
        raise RuntimeError(
            f"Failed to build full K-1 basis: got shape {contrasts.shape}, expected {(k, k - 1)}."
        )
    meta["active_basis_dim"] = int(contrasts.shape[1])
    meta["probabilities"] = probabilities.tolist()
    return categories, contrasts, meta


def transform_discrete_values(
    values: np.ndarray,
    categories: list[float],
    contrast_table: np.ndarray,
) -> np.ndarray:
    """Map discrete values to K-1 contrast coordinates (unknown → zero)."""
    x = np.asarray(values, dtype=float)
    table = np.asarray(contrast_table, dtype=float)
    if table.ndim != 2:
        raise ValueError("contrast_table must be 2-d")
    output = np.zeros((len(x), table.shape[1]), dtype=float)
    if len(categories) == 0 or table.shape[1] == 0:
        return output

    # Vectorized lookup via rounded category codes when categories are numeric.
    cats = np.asarray(categories, dtype=float)
    finite = np.isfinite(x)
    if not np.any(finite):
        return output
    # Match each finite value to a category index (exact float match).
    # Broadcasting: (n, 1) vs (1, K) — fine for small K (binary/count ≤ 20).
    xf = x[finite][:, None]
    matches = xf == cats[None, :]
    has = matches.any(axis=1)
    if not np.any(has):
        return output
    idx = np.argmax(matches, axis=1)
    rows = np.flatnonzero(finite)
    take = rows[has]
    output[take] = table[idx[has]]
    return output


def audit_basis(
    encoded: np.ndarray,
    *,
    mean_atol: float = 1e-6,
    gram_atol: float = 1e-4,
) -> dict:
    """Check E[phi_j]≈0 and Gram≈I on the encoded train matrix."""
    if encoded.size == 0 or encoded.shape[1] == 0:
        return {"ok": True, "means": [], "gram": [], "n": int(encoded.shape[0])}
    means = encoded.mean(axis=0)
    gram = encoded.T @ encoded / max(encoded.shape[0], 1)
    ok_mean = bool(np.allclose(means, 0.0, atol=mean_atol))
    ok_gram = bool(np.allclose(gram, np.eye(encoded.shape[1]), atol=gram_atol))
    return {
        "ok": ok_mean and ok_gram,
        "ok_mean": ok_mean,
        "ok_gram": ok_gram,
        "means": means.tolist(),
        "gram_diag": np.diag(gram).tolist(),
        "n": int(encoded.shape[0]),
        "dim": int(encoded.shape[1]),
    }
