from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from numpy.polynomial.legendre import legval
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


def as_float(values: Sequence[object]) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)


def infer_kind(values: Sequence[object], declared: str | None = None) -> str:
    if declared and str(declared).lower() in {"binary", "count", "continuous", "discrete"}:
        return str(declared).lower()
    arr = as_float(values)
    arr = arr[np.isfinite(arr)]
    unique = np.unique(arr)
    if np.all(np.isin(unique, [0.0, 1.0])):
        return "binary"
    if np.allclose(arr, np.round(arr)):
        return "count"
    return "continuous"


class TrainECDF:
    def fit(self, values: Sequence[object]) -> "TrainECDF":
        arr = as_float(values)
        arr = arr[np.isfinite(arr)]
        if arr.size < 2:
            raise ValueError("ECDF needs at least two finite values")
        self.sorted_ = np.sort(arr)
        return self

    def transform(self, values: Sequence[object]) -> np.ndarray:
        arr = as_float(values)
        n = len(self.sorted_)
        left = np.searchsorted(self.sorted_, arr, side="left")
        right = np.searchsorted(self.sorted_, arr, side="right")
        u = (0.5 * (left + right) + 0.5) / (n + 1.0)
        u[~np.isfinite(arr)] = np.nan
        return np.clip(u, 1e-6, 1 - 1e-6)


def legendre_basis(u: np.ndarray, degree: int = 4) -> np.ndarray:
    x = 2.0 * np.asarray(u, dtype=float) - 1.0
    cols = []
    for j in range(1, degree + 1):
        coeff = np.zeros(j + 1)
        coeff[j] = 1.0
        cols.append(np.sqrt(2 * j + 1.0) * legval(x, coeff))
    return np.column_stack(cols)


class DiscreteBasis:
    """Orthonormal basis under empirical PMF; binary gets one nonconstant basis."""

    def __init__(self, degree: int = 4, smoothing: float = 0.5):
        self.degree = degree
        self.smoothing = smoothing

    def fit(self, values: Sequence[object]) -> "DiscreteBasis":
        arr = as_float(values)
        arr = arr[np.isfinite(arr)]
        support, counts = np.unique(arr, return_counts=True)
        if support.size < 2:
            raise ValueError("Discrete basis needs at least two values")
        p = (counts.astype(float) + self.smoothing)
        p /= p.sum()
        coordinate = np.linspace(-1.0, 1.0, support.size)
        max_degree = min(self.degree, support.size - 1)
        raw = np.column_stack([np.ones(support.size)] + [coordinate**d for d in range(1, max_degree + 1)])
        qcols: list[np.ndarray] = []
        for col in raw.T:
            v = col.copy()
            for q in qcols:
                v -= np.sum(p * v * q) * q
            norm = np.sqrt(np.sum(p * v * v))
            if norm > 1e-10:
                qcols.append(v / norm)
        basis = np.column_stack(qcols)
        self.support_ = support
        self.p_ = p
        self.basis_ = basis[:, 1:]
        return self

    def transform(self, values: Sequence[object]) -> np.ndarray:
        arr = as_float(values)
        out = np.full((len(arr), self.basis_.shape[1]), np.nan)
        for i, value in enumerate(arr):
            if np.isfinite(value):
                idx = int(np.argmin(np.abs(self.support_ - value)))
                out[i] = self.basis_[idx]
        return out


class VariableBasis:
    def __init__(self, kind: str, degree: int = 4, count_threshold: int = 12):
        self.kind = kind
        self.degree = degree
        self.count_threshold = count_threshold

    def fit(self, values: Sequence[object]) -> "VariableBasis":
        arr = as_float(values)
        finite = arr[np.isfinite(arr)]
        unique = np.unique(finite).size
        if self.kind in {"binary", "discrete"} or (self.kind == "count" and unique <= self.count_threshold):
            self.mode = "discrete"
            self.model = DiscreteBasis(self.degree).fit(finite)
        else:
            self.mode = "continuous"
            self.model = TrainECDF().fit(finite)
        return self

    def transform(self, values: Sequence[object]) -> np.ndarray:
        if self.mode == "discrete":
            return self.model.transform(values)
        return legendre_basis(self.model.transform(values), self.degree)


def hcr_matrix(x, y, kind_x: str, kind_y: str, degree: int = 4) -> np.ndarray:
    x = as_float(x)
    y = as_float(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    bx = VariableBasis(kind_x, degree).fit(x)
    by = VariableBasis(kind_y, degree).fit(y)
    fx, fy = bx.transform(x), by.transform(y)
    return fx.T @ fy / len(x)


def hcr_energy(matrix: np.ndarray) -> float:
    return float(np.sum(np.asarray(matrix) ** 2))


def z_design_fit(z_train: pd.DataFrame, z_kinds: Mapping[str, str], degree: int = 2):
    fitted = []
    blocks = []
    for col in z_train.columns:
        tr = VariableBasis(z_kinds[col], degree).fit(z_train[col])
        fitted.append((col, tr))
        blocks.append(tr.transform(z_train[col]))
    if not blocks:
        return fitted, np.ones((len(z_train), 1))
    return fitted, np.column_stack(blocks)


def z_design_transform(z: pd.DataFrame, fitted) -> np.ndarray:
    if not fitted:
        return np.ones((len(z), 1))
    return np.column_stack([tr.transform(z[col]) for col, tr in fitted])


def conditional_pit_continuous(
    target,
    z: pd.DataFrame,
    z_kinds: Mapping[str, str],
    *,
    degree: int = 4,
    ridge_alpha: float = 0.01,
    folds: int = 5,
    seed: int = 20260722,
    epsilon: float = 1e-3,
) -> np.ndarray:
    """Jarek-style cross-fitted conditional CDF for a continuous/high-count target."""
    target = as_float(target)
    valid = np.isfinite(target)
    if z.shape[1]:
        valid &= np.isfinite(z.apply(pd.to_numeric, errors="coerce").to_numpy(float)).all(axis=1)
    indices = np.flatnonzero(valid)
    out = np.full(len(target), np.nan)
    splitter = KFold(folds, shuffle=True, random_state=seed)
    grid = np.linspace(0.0, 1.0, 257)
    grid_basis = legendre_basis(grid, degree)
    dx = grid[1] - grid[0]

    for fold, (fit_local, test_local) in enumerate(splitter.split(indices)):
        fit_idx, test_idx = indices[fit_local], indices[test_local]
        ecdf = TrainECDF().fit(target[fit_idx])
        u_fit = ecdf.transform(target[fit_idx])
        u_test = ecdf.transform(target[test_idx])
        moments_fit = legendre_basis(u_fit, degree)

        z_fit = z.iloc[fit_idx].reset_index(drop=True)
        z_test = z.iloc[test_idx].reset_index(drop=True)
        fitted, design_fit = z_design_fit(z_fit, z_kinds, degree=2)
        design_test = z_design_transform(z_test, fitted)
        model = Ridge(alpha=ridge_alpha).fit(design_fit, moments_fit)
        predicted = model.predict(design_test)

        density = np.maximum(1.0 + predicted @ grid_basis.T, epsilon)
        increments = 0.5 * (density[:, 1:] + density[:, :-1]) * dx
        cdf = np.column_stack([np.zeros(len(test_idx)), np.cumsum(increments, axis=1)])
        cdf /= np.maximum(cdf[:, -1:], epsilon)
        for row, u in enumerate(u_test):
            out[test_idx[row]] = np.interp(u, grid, cdf[row])

    return np.clip(out, 1e-6, 1 - 1e-6)


@dataclass
class ConditioningSelector:
    nodes: pd.DataFrame
    edges: pd.DataFrame
    train_samples: pd.DataFrame
    max_total_z: int = 2

    def __post_init__(self):
        self.nodes = self.nodes.set_index("node", drop=False)
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.nodes.index.astype(str))
        self.graph.add_edges_from(self.edges[["source", "target"]].astype(str).itertuples(index=False, name=None))

    def safe_common_ancestors(self, u: str, v: str, forbidden: set[str]) -> list[str]:
        common = nx.ancestors(self.graph, u) & nx.ancestors(self.graph, v)
        descendants = nx.descendants(self.graph, u) | nx.descendants(self.graph, v)
        result = []
        for node in common:
            if node in forbidden or node in descendants or node not in self.train_samples.columns:
                continue
            meta = self.nodes.loc[node]

            def _flag(value) -> bool:
                if isinstance(value, str):
                    return value.strip().lower() in {"1", "true", "yes"}
                return bool(value)

            if _flag(meta.get("is_endpoint", False)) or _flag(meta.get("is_latent", False)):
                continue
            try:
                obs = float(meta.get("observability", 1.0))
            except (TypeError, ValueError):
                obs = 1.0
            if obs < 0.5:
                continue
            result.append(node)
        return sorted(result)

    def _corr(self, a: str, b: str) -> float:
        value = pd.to_numeric(self.train_samples[a], errors="coerce").corr(
            pd.to_numeric(self.train_samples[b], errors="coerce"), method="spearman"
        )
        return 0.0 if pd.isna(value) else abs(float(value))

    def select(self, u: str, v: str, mandatory: Sequence[str], forbidden: set[str]) -> list[str]:
        selected = [z for z in dict.fromkeys(mandatory) if z in self.train_samples.columns and z not in forbidden]
        pool = [z for z in self.safe_common_ancestors(u, v, forbidden | {u, v}) if z not in selected]
        while pool and len(selected) < self.max_total_z:
            scored = []
            for z in pool:
                relevance = min(self._corr(z, u), self._corr(z, v))
                redundancy = max([self._corr(z, s) for s in selected], default=0.0)
                direct_parent = 1.0 if self.graph.has_edge(z, u) and self.graph.has_edge(z, v) else 0.0
                scored.append((direct_parent + relevance - 0.5 * redundancy, z))
            _, best = max(scored, key=lambda item: (item[0], -ord(item[1][0]) if item[1] else 0))
            selected.append(best)
            pool.remove(best)
        return selected[: self.max_total_z]

    def motif_sets(self, a: str, g: str, b: str, y: str) -> dict[str, list[str]]:
        return {
            "AB": self.select(a, b, mandatory=(), forbidden={g, y}),
            "AY": self.select(a, y, mandatory=(b,), forbidden={g}),
            "BY": self.select(b, y, mandatory=(a,), forbidden={g}),
        }


def mixed_feature_8d(
    x,
    y,
    kind_x: str,
    kind_y: str,
    z: pd.DataFrame | None,
    z_kinds: Mapping[str, str],
    degree: int = 4,
    bootstrap_repeats: int = 20,
    seed: int = 20260722,
) -> np.ndarray:
    """Fixed 8D block for non-binary-binary pairs.

    [first four coefficients, raw energy, conditional energy, support,
     bootstrap SD of raw energy]

    In this first safe implementation, full two-sided conditional PIT is used
    only when both variables are continuous or high-cardinality counts. Mixed
    binary-continuous pairs still receive proper unconditional mixed HCR; their
    conditional extension should be a separate ablation.
    """
    x_arr, y_arr = as_float(x), as_float(y)
    complete = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_complete, y_complete = x_arr[complete], y_arr[complete]

    matrix = hcr_matrix(x_complete, y_complete, kind_x, kind_y, degree)
    flat = matrix.ravel()
    first4 = np.zeros(4)
    first4[: min(4, len(flat))] = flat[:4]
    raw = hcr_energy(matrix)

    if z is not None and z.shape[1] > 0 and kind_x in {"continuous", "count"} and kind_y in {"continuous", "count"}:
        z_complete = z.loc[complete].reset_index(drop=True)
        rx = conditional_pit_continuous(x_complete, z_complete, z_kinds, degree=degree, seed=seed)
        ry = conditional_pit_continuous(y_complete, z_complete, z_kinds, degree=degree, seed=seed + 101)
        mask = np.isfinite(rx) & np.isfinite(ry)
        residual = legendre_basis(rx[mask], degree).T @ legendre_basis(ry[mask], degree) / mask.sum()
        conditional = hcr_energy(residual)
    else:
        conditional = raw

    rng = np.random.default_rng(seed)
    bootstrap_values = []
    for _ in range(max(0, bootstrap_repeats)):
        idx = rng.integers(0, len(x_complete), size=len(x_complete))
        try:
            bootstrap_values.append(
                hcr_energy(hcr_matrix(x_complete[idx], y_complete[idx], kind_x, kind_y, degree))
            )
        except ValueError:
            continue
    uncertainty = float(np.std(bootstrap_values, ddof=1)) if len(bootstrap_values) > 1 else 0.0
    support = float(complete.mean())
    return np.concatenate([first4, [raw, conditional, support, uncertainty]])
