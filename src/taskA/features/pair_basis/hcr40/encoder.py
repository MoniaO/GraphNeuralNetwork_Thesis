"""HCR 40D pair encoder (train-only bases; ordered pairs)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

from taskA.features.pair_basis.variable_spec import VariableType
from taskA.features.pair_basis.variable_specs_v3 import VARIABLE_SPECS
from taskA.features.pair_basis.hcr40.continuous_basis import TrainEmpiricalCDF, shifted_legendre_basis
from taskA.features.pair_basis.hcr40.discrete_basis import fit_discrete_orthonormal_basis, transform_discrete_values

from .jitter import count_to_jittered_u
from .packing import (
    PAIR_DIM,
    legacy_binary_compact_8,
    pack_pair_vector_40,
    pad_coefficient_matrix,
)

DEGREE = 4


def patient_train_fingerprint(train_patients: pd.DataFrame) -> str:
    """Stable hash of patient_train only (never valid/test)."""
    h = hashlib.sha256()
    h.update(f"n={len(train_patients)}".encode("utf-8"))
    for col in sorted(map(str, train_patients.columns)):
        h.update(col.encode("utf-8"))
        values = pd.to_numeric(train_patients[col], errors="coerce").to_numpy(dtype=np.float64)
        h.update(values.tobytes())
    return h.hexdigest()


@dataclass
class Hcr40Config:
    variant: str = "W7B_B3_HYBRID_LEGACY_BINARY_JITTER40"
    pair_dim: int = PAIR_DIM
    motif_dim: int = 120
    min_complete: int = 50
    min_level_count: int = 5
    bootstrap_seed: int = 20260722
    v0_smoothing: float = 0.5
    v1_smoothing: float = 0.0
    count_cap_percentile: float = 99.5
    legendre_degree: int = DEGREE

    @classmethod
    def from_hydra(cls, hcr_cfg: Any, experiment_cfg: Any | None = None) -> "Hcr40Config":
        variant = str(
            getattr(experiment_cfg, "variant", None)
            or getattr(hcr_cfg, "variant", "W7B_B3_HYBRID_LEGACY_BINARY_JITTER40")
        ).strip()
        return cls(
            variant=variant,
            pair_dim=int(getattr(hcr_cfg, "pair_dim", PAIR_DIM)),
            motif_dim=int(getattr(hcr_cfg, "motif_dim", 120)),
            min_complete=int(getattr(hcr_cfg, "min_complete", 50)),
            min_level_count=int(getattr(hcr_cfg, "min_level_count", 5)),
            bootstrap_seed=int(getattr(hcr_cfg, "bootstrap_seed", 20260722)),
            v0_smoothing=float(getattr(hcr_cfg, "v0_smoothing", 0.5)),
            v1_smoothing=float(getattr(hcr_cfg, "v1_smoothing", 0.0)),
            count_cap_percentile=float(getattr(hcr_cfg, "count_cap_percentile", 99.5)),
        )

    def mode(self) -> str:
        v = self.variant.upper()
        if "B0_" in v or v.endswith("PAD40"):
            return "B0"
        if "B1_" in v or "MATRIX40" in v:
            return "B1"
        if "B2_" in v or "ENRICHED40" in v:
            return "B2"
        return "B3"


@dataclass
class FittedVar:
    name: str
    kind: str
    categories: list[float] = field(default_factory=list)
    probabilities: np.ndarray | None = None
    left_cdf: np.ndarray | None = None
    contrasts: np.ndarray | None = None
    ecdf: TrainEmpiricalCDF | None = None
    count_cap: float = 0.0
    dim: int = 0


class Hcr40PairEncoder:
    def __init__(self, config: Hcr40Config) -> None:
        self.config = config
        self.fitted: dict[str, FittedVar] = {}
        self._train_df: pd.DataFrame | None = None
        self.n_train_patients: int = 0
        self.patient_train_fingerprint: str = ""
        self.fit_scope: str = "patient_train_only"
        self.scenario: str = "clean"
        self._pair_cache: dict[tuple, tuple[np.ndarray, dict]] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self._patient_ids: np.ndarray | None = None

    def _kind(self, name: str) -> str:
        spec = VARIABLE_SPECS.get(name)
        if spec is None:
            return "continuous"
        if spec.variable_type == VariableType.BINARY:
            return "binary"
        if spec.variable_type == VariableType.COUNT:
            return "count"
        return "continuous"

    def _column(self, name: str) -> str:
        spec = VARIABLE_SPECS.get(name)
        return spec.column_name if spec is not None else name

    def fit(self, train_patients: pd.DataFrame, variable_names: Iterable[str]) -> "Hcr40PairEncoder":
        self._train_df = train_patients.reset_index(drop=True)
        self.n_train_patients = int(len(self._train_df))
        self.patient_train_fingerprint = patient_train_fingerprint(self._train_df)
        self._pair_cache = {}
        self.cache_hits = self.cache_misses = 0
        if "patient_id" in self._train_df.columns:
            self._patient_ids = self._train_df["patient_id"].astype(str).to_numpy()
        else:
            self._patient_ids = np.arange(self.n_train_patients).astype(str)

        for name in sorted({str(n) for n in variable_names}):
            col = self._column(name)
            if col not in self._train_df.columns:
                continue
            values = pd.to_numeric(self._train_df[col], errors="coerce").to_numpy(dtype=float)
            kind = self._kind(name)
            if kind == "binary":
                cats, contrasts, meta = fit_discrete_orthonormal_basis(
                    values,
                    category_order=[0.0, 1.0],
                    smoothing=self.config.v1_smoothing,
                    max_categories=2,
                )
                self.fitted[name] = FittedVar(
                    name=name,
                    kind=kind,
                    categories=cats,
                    contrasts=contrasts,
                    dim=int(contrasts.shape[1]) if contrasts is not None else 0,
                )
            elif kind == "count":
                finite = values[np.isfinite(values)]
                if finite.size == 0:
                    continue
                cap = float(np.percentile(finite, self.config.count_cap_percentile))
                capped = np.minimum(finite, cap)
                cats = sorted(float(c) for c in np.unique(capped))
                counts = np.array([(capped == c).sum() for c in cats], dtype=float)
                probs = counts / counts.sum()
                left = np.concatenate([[0.0], np.cumsum(probs)[:-1]])
                self.fitted[name] = FittedVar(
                    name=name,
                    kind=kind,
                    categories=cats,
                    probabilities=probs,
                    left_cdf=left,
                    count_cap=cap,
                    dim=self.config.legendre_degree,
                )
            else:
                ecdf = TrainEmpiricalCDF().fit(values)
                self.fitted[name] = FittedVar(
                    name=name,
                    kind=kind,
                    ecdf=ecdf,
                    dim=self.config.legendre_degree,
                )
        return self

    def _encode_phi(self, name: str, values: np.ndarray, row_index: np.ndarray) -> np.ndarray:
        fitted = self.fitted.get(name)
        if fitted is None:
            return np.zeros((len(values), 0), dtype=np.float64)
        if fitted.kind == "binary":
            assert fitted.contrasts is not None
            return transform_discrete_values(values, fitted.categories, fitted.contrasts)

        assert self._patient_ids is not None
        pids = self._patient_ids[row_index]
        if fitted.kind == "count":
            assert fitted.probabilities is not None and fitted.left_cdf is not None
            capped = np.minimum(values, fitted.count_cap)
            u = count_to_jittered_u(
                capped,
                pids,
                variable_name=name,
                seed=self.config.bootstrap_seed,
                categories=fitted.categories,
                probabilities=fitted.probabilities,
                left_cdf=fitted.left_cdf,
            )
            return shifted_legendre_basis(u, self.config.legendre_degree)

        assert fitted.ecdf is not None
        u = fitted.ecdf.transform(values)
        return shifted_legendre_basis(u, self.config.legendre_degree)

    def _coefficient_matrix(self, phi_u: np.ndarray, phi_v: np.ndarray) -> np.ndarray:
        if phi_u.size == 0 or phi_v.size == 0 or phi_u.shape[1] == 0 or phi_v.shape[1] == 0:
            return np.zeros((0, 0), dtype=np.float64)
        return (phi_u.T @ phi_v) / max(phi_u.shape[0], 1)

    def transform_pair(self, u_name: str, v_name: str) -> tuple[np.ndarray, dict]:
        assert self._train_df is not None
        # Ordered-pair cache (orientation preserved — NOT alphabetically sorted).
        key = (
            self.config.variant,
            self.scenario,
            str(u_name),
            str(v_name),
            self.patient_train_fingerprint,
            self.config.mode(),
        )
        if key in self._pair_cache:
            self.cache_hits += 1
            vec, meta = self._pair_cache[key]
            return vec.copy(), dict(meta)
        self.cache_misses += 1
        vec, meta = self._transform_uncached(str(u_name), str(v_name))
        self._pair_cache[key] = (vec, meta)
        return vec.copy(), dict(meta)

    def _transform_uncached(self, u_name: str, v_name: str) -> tuple[np.ndarray, dict]:
        assert self._train_df is not None
        mode = self.config.mode()
        col_u, col_v = self._column(u_name), self._column(v_name)
        kind_u, kind_v = self._kind(u_name), self._kind(v_name)
        n_train = self.n_train_patients

        if col_u not in self._train_df.columns or col_v not in self._train_df.columns:
            z = np.zeros(PAIR_DIM, dtype=np.float32)
            return z, {"supported": False, "reason": "missing_column"}

        u = pd.to_numeric(self._train_df[col_u], errors="coerce").to_numpy(dtype=float)
        v = pd.to_numeric(self._train_df[col_v], errors="coerce").to_numpy(dtype=float)
        complete = np.isfinite(u) & np.isfinite(v)
        n_complete = int(complete.sum())
        idx = np.flatnonzero(complete)

        # B0: only legacy binary–binary; everything else zero-padded types/support
        if mode == "B0":
            if kind_u == "binary" and kind_v == "binary" and n_complete >= self.config.min_complete:
                legacy = legacy_binary_compact_8(u, v, smoothing=self.config.v0_smoothing)
                vec = pack_pair_vector_40(
                    padded_a=np.zeros((4, 4)),
                    d_u=1,
                    d_v=1,
                    kind_u=kind_u,
                    kind_v=kind_v,
                    u_raw=u,
                    v_raw=v,
                    n_complete=n_complete,
                    n_train=n_train,
                    supported=True,
                    mode="B0",
                    legacy8=legacy,
                )
                return vec, {"supported": True, "path": "b0_legacy", "kind_u": kind_u, "kind_v": kind_v}
            vec = pack_pair_vector_40(
                padded_a=np.zeros((4, 4)),
                d_u=0,
                d_v=0,
                kind_u=kind_u,
                kind_v=kind_v,
                u_raw=u,
                v_raw=v,
                n_complete=n_complete,
                n_train=n_train,
                supported=False,
                mode="B0",
            )
            return vec, {"supported": False, "path": "b0_zero", "kind_u": kind_u, "kind_v": kind_v}

        # Support gates
        if n_complete < self.config.min_complete:
            vec = pack_pair_vector_40(
                padded_a=np.zeros((4, 4)),
                d_u=0,
                d_v=0,
                kind_u=kind_u,
                kind_v=kind_v,
                u_raw=u,
                v_raw=v,
                n_complete=n_complete,
                n_train=n_train,
                supported=False,
                mode=mode,
                fill_enrichment=(mode != "B1"),
                fill_matrix=True,
            )
            return vec, {"supported": False, "path": "insufficient_complete"}

        if kind_u == "binary" and kind_v == "binary":
            uu = np.clip(np.round(u[complete]), 0, 1)
            vv = np.clip(np.round(v[complete]), 0, 1)
            levels = min(
                int((uu < 0.5).sum()),
                int((uu >= 0.5).sum()),
                int((vv < 0.5).sum()),
                int((vv >= 0.5).sum()),
            )
            if levels < self.config.min_level_count:
                vec = pack_pair_vector_40(
                    padded_a=np.zeros((4, 4)),
                    d_u=1,
                    d_v=1,
                    kind_u=kind_u,
                    kind_v=kind_v,
                    u_raw=u,
                    v_raw=v,
                    n_complete=n_complete,
                    n_train=n_train,
                    supported=False,
                    mode=mode,
                    fill_enrichment=(mode != "B1"),
                )
                return vec, {"supported": False, "path": "insufficient_levels"}

        phi_u = self._encode_phi(u_name, u[complete], idx)
        phi_v = self._encode_phi(v_name, v[complete], idx)
        d_u = int(phi_u.shape[1]) if phi_u.ndim == 2 else 0
        d_v = int(phi_v.shape[1]) if phi_v.ndim == 2 else 0
        a = self._coefficient_matrix(phi_u, phi_v)
        padded = pad_coefficient_matrix(a, d_u, d_v)
        a11 = float(padded[0, 0]) if padded.size else 0.0

        legacy = None
        if kind_u == "binary" and kind_v == "binary":
            legacy = legacy_binary_compact_8(u, v, smoothing=self.config.v0_smoothing)

        fill_enrichment = mode in {"B2", "B3"}
        # B1: matrix + dependence summaries; types/support still logged in slots 32-39
        if mode == "B1":
            fill_enrichment = False

        fu = self.fitted.get(u_name)
        fv = self.fitted.get(v_name)
        vec = pack_pair_vector_40(
            padded_a=padded,
            d_u=max(d_u, 1),
            d_v=max(d_v, 1),
            kind_u=kind_u,
            kind_v=kind_v,
            u_raw=u,
            v_raw=v,
            n_complete=n_complete,
            n_train=n_train,
            supported=True,
            mode=mode,
            legacy8=legacy,
            a11=a11,
            count_cap_u=float(fu.count_cap) if fu and fu.kind == "count" else 1.0,
            count_cap_v=float(fv.count_cap) if fv and fv.kind == "count" else 1.0,
            fill_enrichment=fill_enrichment,
            fill_matrix=True,
        )

        # B1: zero marginal/joint enrichment but keep dependence + types/support
        if mode == "B1":
            vec = vec.copy()
            vec[20:32] = 0.0

        meta = {
            "supported": True,
            "kind_u": kind_u,
            "kind_v": kind_v,
            "d_u": d_u,
            "d_v": d_v,
            "a11": a11,
            "n_complete": n_complete,
            "mode": mode,
            "path": f"hcr40_{mode.lower()}",
            "fit_scope": self.fit_scope,
            "patient_train_fingerprint": self.patient_train_fingerprint,
        }
        return vec, meta
