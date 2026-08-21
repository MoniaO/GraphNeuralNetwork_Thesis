"""Liczy wektory cech pary S0–S10 (fit wyłącznie na pacjentach train).

Co robi
-------
Z macierzy pacjentów train buduje NMI / Jaccard / cosine / FULL40 (Panel B).
`slice_full40` wycina prefiksy S5–S10 z tego samego wektora 40D.

Co wolno zmieniać
-----------------
Nowy rodzaj cechy = nowy wariant w variants.py + funkcja tutaj.
Nie zmieniaj kolejności slotów FULL40 (HCR16, energy, marg, joint, META8).

Czego nie ruszać dla FINAL
--------------------------
Fit na train, zero G_true, zero cichej binaryzacji w S4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

from taskA.features.pair_basis.variable_spec import VariableType
from taskA.features.pair_basis.variable_specs_v3 import VARIABLE_SPECS
from taskA.features.pair_basis.wave7.panel_b.encoder import PanelBConfig, PanelBPairEncoder
from taskA.features.pair_basis.wave7.panel_b.packing import PAIR_DIM
from taskA.features.pair_basis.wave7.panel_b.summaries import active_mask, continuous_marginal

from .variants import StageCVariant, get_variant

EPS = 1e-12
B2_VARIANT = "W7B_B2_JITTER_GHCR_ENRICHED40"
NMI_MAX_BINS = 8


def _kind(name: str) -> str:
    spec = VARIABLE_SPECS.get(str(name))
    if spec is None:
        return "continuous"
    if spec.variable_type == VariableType.BINARY:
        return "binary"
    if spec.variable_type == VariableType.COUNT:
        return "count"
    return "continuous"


def _column(name: str) -> str:
    spec = VARIABLE_SPECS.get(str(name))
    return spec.column_name if spec is not None else str(name)


def _values(train_df: pd.DataFrame, name: str) -> np.ndarray | None:
    col = _column(name)
    if col not in train_df.columns:
        return None
    return pd.to_numeric(train_df[col], errors="coerce").to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# NMI (train-only discretization)
# ---------------------------------------------------------------------------


@dataclass
class _Discretizer:
    kind: str
    edges: np.ndarray | None = None  # quantile edges for continuous/count
    levels: np.ndarray | None = None  # exact count levels when n_unique <= 8

    def transform(self, x: np.ndarray) -> np.ndarray:
        xf = np.asarray(x, dtype=float)
        out = np.full(len(xf), -1, dtype=np.int64)
        finite = np.isfinite(xf)
        if not finite.any():
            return out
        if self.kind == "binary":
            out[finite] = (xf[finite] >= 0.5).astype(np.int64)
            return out
        if self.levels is not None:
            # map exact levels; unknowns → -1
            level_to_i = {float(v): i for i, v in enumerate(self.levels)}
            for i in np.flatnonzero(finite):
                out[i] = level_to_i.get(float(xf[i]), -1)
            return out
        assert self.edges is not None
        # digitize into [0, n_bins-1]; edges are interior cut points
        bins = np.digitize(xf[finite], self.edges, right=False)
        n_bins = len(self.edges) + 1
        out[finite] = np.clip(bins, 0, n_bins - 1)
        return out


def _fit_discretizer(values: np.ndarray, kind: str) -> _Discretizer:
    x = np.asarray(values, dtype=float)
    finite = x[np.isfinite(x)]
    if kind == "binary":
        return _Discretizer(kind="binary")
    if kind == "count":
        if finite.size == 0:
            return _Discretizer(kind="count", levels=np.array([0.0]))
        uniq = np.unique(finite)
        if len(uniq) <= NMI_MAX_BINS:
            return _Discretizer(kind="count", levels=uniq.astype(float))
        # train-fitted quantile bins, max 8
        qs = np.linspace(0, 1, NMI_MAX_BINS + 1)[1:-1]
        edges = np.unique(np.quantile(finite, qs))
        return _Discretizer(kind="count", edges=edges)
    # continuous: 8 quantile bins
    if finite.size < 2:
        return _Discretizer(kind="continuous", edges=np.array([]))
    qs = np.linspace(0, 1, NMI_MAX_BINS + 1)[1:-1]
    edges = np.unique(np.quantile(finite, qs))
    return _Discretizer(kind="continuous", edges=edges)


def normalized_mutual_information(
    u: np.ndarray,
    v: np.ndarray,
    *,
    disc_u: _Discretizer,
    disc_v: _Discretizer,
) -> tuple[float, float]:
    """Return (NMI, RAW_MI) on complete rows; train discretizers already fitted."""
    uu = disc_u.transform(u)
    vv = disc_v.transform(v)
    complete = (uu >= 0) & (vv >= 0)
    if complete.sum() < 2:
        return 0.0, 0.0
    a = uu[complete]
    b = vv[complete]
    n = int(len(a))
    # joint counts
    a_levels = np.unique(a)
    b_levels = np.unique(b)
    mi = 0.0
    p_a = {int(k): float(np.mean(a == k)) for k in a_levels}
    p_b = {int(k): float(np.mean(b == k)) for k in b_levels}
    for ia in a_levels:
        for ib in b_levels:
            p_ab = float(np.mean((a == ia) & (b == ib)))
            if p_ab <= 0:
                continue
            mi += p_ab * np.log(p_ab / max(p_a[int(ia)] * p_b[int(ib)], EPS))
    h_u = -sum(p * np.log(max(p, EPS)) for p in p_a.values() if p > 0)
    h_v = -sum(p * np.log(max(p, EPS)) for p in p_b.values() if p > 0)
    denom = np.sqrt(h_u * h_v)
    nmi = float(mi / denom) if denom > 0 else 0.0
    return float(nmi), float(mi)


# ---------------------------------------------------------------------------
# Activity sets → Jaccard / Cosine
# ---------------------------------------------------------------------------


def _activity_indicator(values: np.ndarray, kind: str) -> np.ndarray:
    """Train-only activity; continuous uses audited robust-z (|z|>1.5)."""
    x = np.asarray(values, dtype=float)
    if kind == "binary":
        return active_mask(x, "binary")
    if kind == "count":
        return active_mask(x, "count")
    _, rz = continuous_marginal(x)
    return active_mask(x, "continuous", robust_z=rz)


def jaccard_active(u: np.ndarray, v: np.ndarray, *, kind_u: str, kind_v: str) -> float:
    complete = np.isfinite(u) & np.isfinite(v)
    if not complete.any():
        return 0.0
    au = _activity_indicator(u, kind_u) & complete
    av = _activity_indicator(v, kind_v) & complete
    inter = int(np.sum(au & av))
    union = int(np.sum(au | av))
    if union == 0:
        return 0.0
    return float(inter / union)


def cosine_active(u: np.ndarray, v: np.ndarray, *, kind_u: str, kind_v: str) -> float:
    complete = np.isfinite(u) & np.isfinite(v)
    if not complete.any():
        return 0.0
    au = _activity_indicator(u, kind_u) & complete
    av = _activity_indicator(v, kind_v) & complete
    inter = int(np.sum(au & av))
    nu = int(np.sum(au))
    nv = int(np.sum(av))
    denom = np.sqrt(float(nu) * float(nv))
    if denom <= 0:
        return 0.0
    return float(inter / denom)


# ---------------------------------------------------------------------------
# HCR_BINARY_ONLY — signed Pearson phi (no smoothing, no silent binarize)
# ---------------------------------------------------------------------------


def signed_phi_binary(u: np.ndarray, v: np.ndarray) -> tuple[float, bool]:
    """Signed a11 / Pearson phi on complete rows. applicable only if both binary-valued."""
    mask = np.isfinite(u) & np.isfinite(v)
    if mask.sum() < 2:
        return 0.0, False
    uu = u[mask]
    vv = v[mask]
    # Strict binary check — do NOT threshold continuous/count.
    uniq_u = set(np.unique(np.round(uu, 10)).tolist())
    uniq_v = set(np.unique(np.round(vv, 10)).tolist())
    if not uniq_u.issubset({0.0, 1.0}) or not uniq_v.issubset({0.0, 1.0}):
        return 0.0, False
    x = (uu >= 0.5).astype(np.int64)
    y = (vv >= 0.5).astype(np.int64)
    n = int(len(x))
    n11 = int(np.sum((x == 1) & (y == 1)))
    n1dot = int(np.sum(x == 1))
    n0dot = n - n1dot
    ndot1 = int(np.sum(y == 1))
    ndot0 = n - ndot1
    denom = np.sqrt(float(n1dot) * float(n0dot) * float(ndot1) * float(ndot0))
    if denom <= 0:
        return 0.0, True
    a11 = (n * n11 - n1dot * ndot1) / denom
    return float(a11), True


# ---------------------------------------------------------------------------
# HCR FULL40 slices via Panel B
# ---------------------------------------------------------------------------


def slice_full40(vec40: np.ndarray, variant: StageCVariant) -> np.ndarray:
    v = np.asarray(vec40, dtype=np.float32).reshape(-1)
    if v.shape[0] != PAIR_DIM:
        raise ValueError(f"expected FULL40, got {v.shape}")
    hcr16 = v[0:16]
    energy4 = v[16:20]
    marg_u = v[20:24]
    marg_v = v[24:28]
    joint4 = v[28:32]
    meta8 = v[32:40]
    name = variant.name
    if name == "META8":
        return meta8.copy()
    if name == "HCR_MATRIX16":
        return hcr16.copy()
    if name == "HCR_COMPACT24":
        return np.concatenate([hcr16, meta8]).astype(np.float32)
    if name == "HCR_COMPACT32":
        return np.concatenate([hcr16, marg_u, marg_v, meta8]).astype(np.float32)
    if name == "HCR_COMPACT36":
        return np.concatenate([hcr16, marg_u, marg_v, joint4, meta8]).astype(np.float32)
    if name == "HCR_FULL40":
        return v.copy()
    raise ValueError(f"not an HCR-slice variant: {variant.config_id}")


@dataclass
class StageCFeatureStore:
    """Fitted train-only pair statistics for one Stage C variant."""

    variant: StageCVariant
    pair_raw: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    pair_applicable: dict[tuple[str, str], bool] = field(default_factory=dict)
    pair_meta: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    patient_train_fingerprint: str = ""
    n_train_patients: int = 0
    raw_mi_audit: dict[tuple[str, str], float] = field(default_factory=dict)

    @property
    def raw_dim(self) -> int:
        # S0 uses placeholder dim 1 of zeros for tensor shapes; decoder bypasses.
        return max(int(self.variant.raw_dim), 1)


def fit_stage_c_features(
    train_patients: pd.DataFrame,
    pairs: Iterable[tuple[str, str]],
    variant: StageCVariant | str,
    *,
    scenario: str = "clean",
) -> StageCFeatureStore:
    """Fit all required ordered-pair features on train patients only."""
    variant = get_variant(variant) if isinstance(variant, str) else variant
    train_df = train_patients.reset_index(drop=True)
    store = StageCFeatureStore(variant=variant)
    store.n_train_patients = int(len(train_df))

    unique_pairs = sorted({(str(u), str(v)) for u, v in pairs})
    unique_nodes = sorted({n for uv in unique_pairs for n in uv})

    # Panel B encoder for HCR slices (reuse Legendre / jitter internals).
    panel: PanelBPairEncoder | None = None
    if variant.kind == "hcr_slice":
        cfg = PanelBConfig(variant=B2_VARIANT)
        panel = PanelBPairEncoder(cfg)
        panel.scenario = scenario
        panel.fit(train_df, unique_nodes)
        store.patient_train_fingerprint = panel.patient_train_fingerprint

    # NMI discretizers (train-only).
    disc: dict[str, _Discretizer] = {}
    if variant.name == "NMI":
        for name in unique_nodes:
            vals = _values(train_df, name)
            if vals is None:
                continue
            disc[name] = _fit_discretizer(vals, _kind(name))

    from taskA.features.pair_basis.wave7.panel_b.encoder import patient_train_fingerprint

    if not store.patient_train_fingerprint:
        store.patient_train_fingerprint = patient_train_fingerprint(train_df)

    d = store.raw_dim
    for u_name, v_name in unique_pairs:
        key = (u_name, v_name)
        kind_u, kind_v = _kind(u_name), _kind(v_name)
        u = _values(train_df, u_name)
        v = _values(train_df, v_name)
        meta: dict[str, Any] = {"kind_u": kind_u, "kind_v": kind_v}

        if variant.kind == "none":
            store.pair_raw[key] = np.zeros(d, dtype=np.float32)
            store.pair_applicable[key] = True
            store.pair_meta[key] = {**meta, "path": "s0_none"}
            continue

        if u is None or v is None:
            store.pair_raw[key] = np.zeros(d, dtype=np.float32)
            store.pair_applicable[key] = False
            store.pair_meta[key] = {**meta, "path": "missing_column"}
            continue

        if variant.name == "NMI":
            du = disc.get(u_name) or _fit_discretizer(u, kind_u)
            dv = disc.get(v_name) or _fit_discretizer(v, kind_v)
            nmi, raw_mi = normalized_mutual_information(u, v, disc_u=du, disc_v=dv)
            store.pair_raw[key] = np.array([nmi], dtype=np.float32)
            store.pair_applicable[key] = True
            store.raw_mi_audit[key] = raw_mi
            store.pair_meta[key] = {**meta, "path": "nmi", "raw_mi": raw_mi}
            continue

        if variant.name == "JACCARD_ACTIVE":
            j = jaccard_active(u, v, kind_u=kind_u, kind_v=kind_v)
            store.pair_raw[key] = np.array([j], dtype=np.float32)
            store.pair_applicable[key] = True
            store.pair_meta[key] = {**meta, "path": "jaccard_active"}
            continue

        if variant.name == "COSINE_ACTIVE":
            c = cosine_active(u, v, kind_u=kind_u, kind_v=kind_v)
            store.pair_raw[key] = np.array([c], dtype=np.float32)
            store.pair_applicable[key] = True
            store.pair_meta[key] = {**meta, "path": "cosine_active"}
            continue

        if variant.name == "HCR_BINARY_ONLY":
            # Applicable ONLY when both variables are typed binary — never binarize others.
            if kind_u != "binary" or kind_v != "binary":
                store.pair_raw[key] = np.zeros(1, dtype=np.float32)
                store.pair_applicable[key] = False
                store.pair_meta[key] = {**meta, "path": "non_binary_masked"}
                continue
            phi, ok = signed_phi_binary(u, v)
            store.pair_raw[key] = np.array([phi], dtype=np.float32)
            store.pair_applicable[key] = bool(ok)
            store.pair_meta[key] = {**meta, "path": "signed_phi", "applicable": ok}
            continue

        if variant.kind == "hcr_slice":
            assert panel is not None
            vec40, pmeta = panel.transform_pair(u_name, v_name)
            sliced = slice_full40(vec40, variant)
            assert sliced.shape[0] == variant.raw_dim
            store.pair_raw[key] = sliced.astype(np.float32)
            store.pair_applicable[key] = True
            store.pair_meta[key] = {**meta, **pmeta, "path": f"panel_b_slice_{variant.name}"}
            continue

        raise ValueError(f"Unhandled variant {variant.config_id}")

    return store
