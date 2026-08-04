"""Train-only empirical CDF + shifted normalized Legendre basis (degrees 1..D)."""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import legval


class TrainEmpiricalCDF:
    def fit(self, values: np.ndarray) -> "TrainEmpiricalCDF":
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 2:
            raise ValueError("Empirical CDF needs at least two finite train values.")
        self.sorted_ = np.sort(arr)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        n = len(self.sorted_)
        left = np.searchsorted(self.sorted_, arr, side="left")
        right = np.searchsorted(self.sorted_, arr, side="right")
        u = (0.5 * (left + right) + 0.5) / (n + 1.0)
        out = np.clip(u, 1e-6, 1.0 - 1e-6)
        out[~np.isfinite(arr)] = np.nan
        return out


def shifted_legendre_basis(u: np.ndarray, degree: int = 4) -> np.ndarray:
    """Normalized Legendre polynomials of degrees 1..degree on [-1,1] after u→x map."""
    u = np.asarray(u, dtype=float)
    x = 2.0 * u - 1.0
    cols = []
    for j in range(1, degree + 1):
        coeff = np.zeros(j + 1, dtype=float)
        coeff[j] = 1.0
        cols.append(np.sqrt(2 * j + 1.0) * legval(x, coeff))
    mat = np.column_stack(cols)
    mat[~np.isfinite(u)] = 0.0
    return mat
