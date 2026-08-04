"""Wave 7 generalized HCR pair encoder (train-only fit)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Hashable, Iterable

import numpy as np
import pandas as pd

from hcr.binary_features import binary_pair_features, result_to_vector
from hcr.variable_spec import VariableType
from hcr.variable_specs_v3 import VARIABLE_SPECS

from .continuous_basis import TrainEmpiricalCDF, shifted_legendre_basis
from .discrete_basis import audit_basis, fit_discrete_orthonormal_basis, transform_discrete_values
from .mapping import PAIR_DIM, coefficient_stats, pack_pair_vector, pair_kinds
from .motif import PAIR_ROLES
from .v1_bootstrap import (
    _a11_binary_with_smoothing,
    _bootstrap_a11_stats_vectorized,
    _bootstrap_sd_a11_reference,
    canonical_pair_names,
    v1_bootstrap_seed,
)

# Re-export reference for tests.
__all_ref__ = ("_bootstrap_sd_a11_reference",)


@dataclass
class Wave7HCRConfig:
    variant: str = "W7_V2_GHCR_ALL_TYPES_ONEHOT_COUNT"
    pair_dim: int = PAIR_DIM
    motif_dim: int = 24
    roles: tuple[str, ...] = PAIR_ROLES
    smoothing: float = 0.5
    continuous_degree: int = 4
    max_categories: int = 20
    min_complete: int = 50
    min_level_count: int = 5
    bootstrap_repeats: int = 30
    bootstrap_seed: int = 20260722
    unknown_category_policy: str = "zero"
    drop_constant_direction: bool = True
    # Clean V1 GHCR: empirical frequencies, no additive smoothing.
    v1_smoothing: float = 0.0

    @classmethod
    def from_hydra(cls, hcr_cfg: Any, experiment_cfg: Any | None = None) -> "Wave7HCRConfig":
        variant = str(
            getattr(experiment_cfg, "variant", None)
            or getattr(hcr_cfg, "variant", "W7_V2_GHCR_ALL_TYPES_ONEHOT_COUNT")
        ).strip()
        binary = getattr(hcr_cfg, "binary", None)
        count = getattr(hcr_cfg, "count", None)
        continuous = getattr(hcr_cfg, "continuous", None)
        uncertainty = getattr(hcr_cfg, "uncertainty", None)
        return cls(
            variant=variant,
            pair_dim=int(getattr(hcr_cfg, "pair_dim", PAIR_DIM)),
            motif_dim=int(getattr(hcr_cfg, "motif_dim", 24)),
            roles=tuple(getattr(hcr_cfg, "roles", list(PAIR_ROLES))),
            smoothing=float(
                getattr(binary, "smoothing", None)
                or getattr(count, "smoothing", None)
                or getattr(hcr_cfg, "smoothing", 0.5)
            ),
            continuous_degree=int(getattr(continuous, "degree", 4) if continuous is not None else 4),
            max_categories=int(getattr(count, "max_categories", 20) if count is not None else 20),
            min_complete=int(getattr(hcr_cfg, "min_complete", 50)),
            min_level_count=int(getattr(hcr_cfg, "min_level_count", 5)),
            bootstrap_repeats=int(
                getattr(uncertainty, "repeats", 30) if uncertainty is not None else 30
            ),
            bootstrap_seed=int(getattr(hcr_cfg, "bootstrap_seed", 20260722)),
            v1_smoothing=float(getattr(hcr_cfg, "v1_smoothing", 0.0)),
        )


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
class FittedVariable:
    name: str
    kind: str
    mode: str  # discrete | continuous | legacy_binary | unsupported
    categories: list[float] = field(default_factory=list)
    contrasts: np.ndarray | None = None
    ecdf: TrainEmpiricalCDF | None = None
    degree: int = 4
    audit: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def _binary_level_counts(u: np.ndarray, v: np.ndarray) -> tuple[int, int, int, int]:
    uu = (np.asarray(u, dtype=float) >= 0.5).astype(np.int64)
    vv = (np.asarray(v, dtype=float) >= 0.5).astype(np.int64)
    n_u0 = int((uu == 0).sum())
    n_u1 = int((uu == 1).sum())
    n_v0 = int((vv == 0).sum())
    n_v1 = int((vv == 1).sum())
    return n_u0, n_u1, n_v0, n_v1


def _fit_binary_contrast(values: np.ndarray, *, smoothing: float) -> tuple[list[float], np.ndarray, dict]:
    """Empirical one-hot → Gram–Schmidt contrast on a pair-complete sample."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    # Force declared {0,1}; with min_level_count both levels are present → no smoothing.
    return fit_discrete_orthonormal_basis(
        x,
        category_order=[0.0, 1.0],
        smoothing=float(smoothing),
        max_categories=2,
    )


def _a11_from_binary_sample(u: np.ndarray, v: np.ndarray, *, smoothing: float) -> float:
    """a11 = (1/n) Σ φ_U(U_i) φ_V(V_i) with bases refit on this sample."""
    uu = np.clip(np.round(np.asarray(u, dtype=np.float64)), 0, 1)
    vv = np.clip(np.round(np.asarray(v, dtype=np.float64)), 0, 1)
    return _a11_binary_with_smoothing(uu, vv, smoothing=float(smoothing))


class Wave7PairEncoder:
    """Fit type-specific bases on train patients; emit fixed 8-d pair vectors."""

    def __init__(self, config: Wave7HCRConfig) -> None:
        self.config = config
        self.fitted: dict[str, FittedVariable] = {}
        self.basis_audit: dict[str, dict] = {}
        self.n_train_patients: int = 0
        self._train_df: pd.DataFrame | None = None
        self._pair_cache: dict[Hashable, tuple[np.ndarray, dict]] = {}
        self.scenario: str = "clean"
        self.fit_scope: str = "patient_train_only"
        self.patient_train_fingerprint: str = ""
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.required_pair_calls: int = 0

    def _kind(self, name: str) -> str:
        spec = VARIABLE_SPECS.get(name)
        if spec is None:
            return "continuous"
        t = spec.variable_type
        if t == VariableType.BINARY:
            return "binary"
        if t == VariableType.COUNT:
            return "count"
        return "continuous"

    def _column(self, name: str) -> str:
        spec = VARIABLE_SPECS.get(name)
        return spec.column_name if spec is not None else name

    def _is_v0(self) -> bool:
        return self.config.variant.upper().endswith("L2_BINARY_COMPACT")

    def _is_v2(self) -> bool:
        return self.config.variant.upper().endswith("ALL_TYPES_ONEHOT_COUNT")

    def _is_v1(self) -> bool:
        """Pure GHCR dependence-only (not V1B matched-aux)."""
        v = self.config.variant.upper()
        return (
            "GHCR_BINARY_ONEHOT" in v
            and "MATCHED_AUX" not in v
            and "V1B" not in v
            and not self._is_v2()
        )

    def _is_v1b(self) -> bool:
        """GHCR a11 + V0 joint prevalence / rarity auxiliaries."""
        v = self.config.variant.upper()
        return "MATCHED_AUX" in v or "V1B_GHCR" in v

    def _v1_cache_key(self, u_name: str, v_name: str) -> tuple:
        left, right = canonical_pair_names(u_name, v_name)
        return (
            str(self.config.variant),
            str(self.scenario),
            left,
            right,
            str(self.patient_train_fingerprint),
            int(self.config.bootstrap_seed),
            int(self.config.bootstrap_repeats),
            float(self.config.v1_smoothing),
            int(self.config.min_complete),
            int(self.config.min_level_count),
        )

    def fit(self, train_patients: pd.DataFrame, variable_names: Iterable[str]) -> "Wave7PairEncoder":
        self._train_df = train_patients.reset_index(drop=True)
        self.n_train_patients = int(len(self._train_df))
        self.patient_train_fingerprint = patient_train_fingerprint(self._train_df)
        self.fit_scope = "patient_train_only"
        self._pair_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.required_pair_calls = 0
        assert self.fit_scope == "patient_train_only"
        assert self.patient_train_fingerprint, "missing patient_train_fingerprint"
        names = sorted({str(n) for n in variable_names})
        for name in names:
            col = self._column(name)
            if col not in self._train_df.columns:
                continue
            values = pd.to_numeric(self._train_df[col], errors="coerce").to_numpy(dtype=float)
            kind = self._kind(name)

            if self._is_v0():
                self.fitted[name] = FittedVariable(name=name, kind=kind, mode="legacy_binary")
                continue

            # V1 / V1B: binary GHCR is fit pair-wise on I_UV at transform time.
            if self._is_v1() or self._is_v1b():
                if kind == "binary":
                    self.fitted[name] = FittedVariable(name=name, kind=kind, mode="v1_pairwise")
                else:
                    self.fitted[name] = FittedVariable(name=name, kind=kind, mode="unsupported")
                continue

            fit_discrete = kind == "binary" or (kind == "count" and self._is_v2())
            fit_continuous = kind == "continuous" and self._is_v2()

            if fit_discrete:
                max_cat = self.config.max_categories if kind == "count" else 2
                try:
                    categories, contrasts, meta = fit_discrete_orthonormal_basis(
                        values,
                        category_order=[0.0, 1.0] if kind == "binary" else None,
                        smoothing=self.config.smoothing,
                        max_categories=max_cat,
                    )
                except ValueError as exc:
                    raise ValueError(f"Wave7 basis fit failed for '{name}' ({kind}): {exc}") from exc
                encoded = transform_discrete_values(values[np.isfinite(values)], categories, contrasts)
                audit = audit_basis(encoded)
                self.basis_audit[name] = {
                    "variable": name,
                    "kind": kind,
                    "observed_categories": categories,
                    "K": meta["K"],
                    "active_basis_dim": meta["active_basis_dim"],
                    "min_category_count": meta["min_category_count"],
                    "used_smoothing": meta["used_smoothing"],
                    **audit,
                }
                self.fitted[name] = FittedVariable(
                    name=name,
                    kind=kind,
                    mode="discrete",
                    categories=categories,
                    contrasts=contrasts,
                    audit=audit,
                    meta=meta,
                )
            elif fit_continuous:
                ecdf = TrainEmpiricalCDF().fit(values)
                u = ecdf.transform(values)
                phi = shifted_legendre_basis(u[np.isfinite(u)], self.config.continuous_degree)
                audit = audit_basis(phi, mean_atol=5e-2, gram_atol=5e-2)
                self.basis_audit[name] = {
                    "variable": name,
                    "kind": kind,
                    "K": None,
                    "active_basis_dim": int(phi.shape[1]),
                    **audit,
                }
                self.fitted[name] = FittedVariable(
                    name=name,
                    kind=kind,
                    mode="continuous",
                    ecdf=ecdf,
                    degree=self.config.continuous_degree,
                    audit=audit,
                    meta={"active_basis_dim": int(phi.shape[1])},
                )
            else:
                self.fitted[name] = FittedVariable(name=name, kind=kind, mode="unsupported")
        return self

    def _encode_series(self, name: str, values: np.ndarray) -> tuple[np.ndarray, str]:
        fitted = self.fitted.get(name)
        kind = self._kind(name)
        if fitted is None or fitted.mode in {"unsupported", "legacy_binary", "v1_pairwise"}:
            return np.zeros((len(values), 0), dtype=float), kind
        if fitted.mode == "discrete":
            assert fitted.contrasts is not None
            return transform_discrete_values(values, fitted.categories, fitted.contrasts), kind
        assert fitted.ecdf is not None
        u = fitted.ecdf.transform(values)
        return shifted_legendre_basis(u, fitted.degree), kind

    def _legacy_binary_compact(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        mask = np.isfinite(u) & np.isfinite(v)
        if mask.sum() < 2:
            return np.zeros(PAIR_DIM, dtype=np.float32)
        uu = np.clip(np.round(u[mask]), 0, 1).astype(np.int64)
        vv = np.clip(np.round(v[mask]), 0, 1).astype(np.int64)
        if not (set(np.unique(uu)).issubset({0, 1}) and set(np.unique(vv)).issubset({0, 1})):
            return np.zeros(PAIR_DIM, dtype=np.float32)
        result = binary_pair_features(uu, vv, smoothing=self.config.smoothing)
        return result_to_vector(result).astype(np.float32)

    def _coefficient_matrix(self, phi_u: np.ndarray, phi_v: np.ndarray) -> np.ndarray:
        if phi_u.size == 0 or phi_v.size == 0 or phi_u.shape[1] == 0 or phi_v.shape[1] == 0:
            return np.zeros((0, 0), dtype=float)
        return phi_u.T @ phi_v / max(phi_u.shape[0], 1)

    def _transform_v1_binary_pair(
        self, u: np.ndarray, v: np.ndarray, *, u_name: str, v_name: str
    ) -> tuple[np.ndarray, dict]:
        """Clean V1 GHCR on pair-complete sample I_UV (patient_train only)."""
        complete = np.isfinite(u) & np.isfinite(v)
        n_complete = int(complete.sum())
        n_train = int(self.n_train_patients)
        assert n_train == int(len(u)), "V1 must use patient_train rows only"
        support = float(n_complete / max(n_train, 1))

        uu = np.clip(np.round(u[complete]), 0, 1).astype(np.float64)
        vv = np.clip(np.round(v[complete]), 0, 1).astype(np.float64)
        n_u0, n_u1, n_v0, n_v1 = _binary_level_counts(uu, vv)
        min_level_count = int(min(n_u0, n_u1, n_v0, n_v1))

        if n_complete < self.config.min_complete or min_level_count < self.config.min_level_count:
            return np.zeros(PAIR_DIM, dtype=np.float32), {
                "supported": False,
                "kind_u": "binary",
                "kind_v": "binary",
                "support": support,
                "n_complete": n_complete,
                "min_level_count": min_level_count,
                "n_u0": n_u0,
                "n_u1": n_u1,
                "n_v0": n_v0,
                "n_v1": n_v1,
                "path": "insufficient_support",
                "fit_scope": self.fit_scope,
                "patient_train_fingerprint": self.patient_train_fingerprint,
            }

        a11 = float(_a11_from_binary_sample(uu, vv, smoothing=self.config.v1_smoothing))
        energy = float(a11**2)

        boot_seed = v1_bootstrap_seed(
            global_seed=int(self.config.bootstrap_seed),
            scenario=str(self.scenario),
            u_name=u_name,
            v_name=v_name,
        )
        bootstrap_sd_a11, bootstrap_sd_energy, n_valid = _bootstrap_a11_stats_vectorized(
            uu,
            vv,
            repeats=int(self.config.bootstrap_repeats),
            smoothing=float(self.config.v1_smoothing),
            seed=boot_seed,
        )

        vec = np.zeros(PAIR_DIM, dtype=np.float32)
        vec[0] = np.float32(a11)
        vec[4] = np.float32(energy)
        vec[5] = np.float32(support)
        vec[6] = np.float32(bootstrap_sd_a11)
        vec[7] = np.float32(1.0)

        return vec, {
            "supported": True,
            "kind_u": "binary",
            "kind_v": "binary",
            "support": support,
            "n_complete": n_complete,
            "n_train": n_train,
            "min_level_count": min_level_count,
            "n_u0": n_u0,
            "n_u1": n_u1,
            "n_v0": n_v0,
            "n_v1": n_v1,
            "a11": a11,
            "energy": energy,
            "bootstrap_sd_a11": bootstrap_sd_a11,
            "bootstrap_sd_energy": bootstrap_sd_energy,
            "bootstrap_repeats_requested": int(self.config.bootstrap_repeats),
            "bootstrap_repeats_valid": int(n_valid),
            "bootstrap_seed": int(boot_seed),
            "used_smoothing_u": float(self.config.v1_smoothing) != 0.0,
            "used_smoothing_v": float(self.config.v1_smoothing) != 0.0,
            "fit_scope": self.fit_scope,
            "patient_train_fingerprint": self.patient_train_fingerprint,
            "path": "ghcr_v1_pairwise",
        }

    def _transform_v1b_binary_pair(
        self, u: np.ndarray, v: np.ndarray, *, u_name: str, v_name: str
    ) -> tuple[np.ndarray, dict]:
        """V1B: GHCR a11 + V0-matched joint prevalence / rarity auxiliaries.

        Vector:
          [a11, 0, 0, 0, a11², p11=n11/n_complete, 1/√(n11+1), mask]

        Bootstrap SD is audit-only (not placed in the network vector).
        """
        complete = np.isfinite(u) & np.isfinite(v)
        n_complete = int(complete.sum())
        n_train = int(self.n_train_patients)
        assert n_train == int(len(u)), "V1B must use patient_train rows only"
        completeness = float(n_complete / max(n_train, 1))

        uu = np.clip(np.round(u[complete]), 0, 1).astype(np.float64)
        vv = np.clip(np.round(v[complete]), 0, 1).astype(np.float64)
        n_u0, n_u1, n_v0, n_v1 = _binary_level_counts(uu, vv)
        min_level_count = int(min(n_u0, n_u1, n_v0, n_v1))

        if n_complete < self.config.min_complete or min_level_count < self.config.min_level_count:
            return np.zeros(PAIR_DIM, dtype=np.float32), {
                "supported": False,
                "kind_u": "binary",
                "kind_v": "binary",
                "path": "insufficient_support",
                "n_complete": n_complete,
                "min_level_count": min_level_count,
                "completeness": completeness,
                "fit_scope": self.fit_scope,
                "patient_train_fingerprint": self.patient_train_fingerprint,
            }

        a11 = float(_a11_from_binary_sample(uu, vv, smoothing=self.config.v1_smoothing))
        energy = float(a11**2)
        n11 = int(((uu >= 0.5) & (vv >= 0.5)).sum())
        # Joint prevalence (NOT completeness). Same raw definition as V0 joint_support_rate.
        p11 = float(n11 / max(n_complete, 1))
        rarity = float(1.0 / np.sqrt(n11 + 1.0))

        boot_seed = v1_bootstrap_seed(
            global_seed=int(self.config.bootstrap_seed),
            scenario=str(self.scenario),
            u_name=u_name,
            v_name=v_name,
        )
        bootstrap_sd_a11, bootstrap_sd_energy, n_valid = _bootstrap_a11_stats_vectorized(
            uu,
            vv,
            repeats=int(self.config.bootstrap_repeats),
            smoothing=float(self.config.v1_smoothing),
            seed=boot_seed,
        )

        vec = np.zeros(PAIR_DIM, dtype=np.float32)
        vec[0] = np.float32(a11)
        vec[4] = np.float32(energy)
        vec[5] = np.float32(p11)  # joint prevalence
        vec[6] = np.float32(rarity)  # V0-matched rarity proxy
        vec[7] = np.float32(1.0)

        return vec, {
            "supported": True,
            "kind_u": "binary",
            "kind_v": "binary",
            "a11": a11,
            "energy": energy,
            "joint_prevalence_p11": p11,
            "legacy_rarity": rarity,
            "n11": n11,
            "n_complete": n_complete,
            "completeness_n_complete_over_n_train": completeness,
            "p_u": float(uu.mean()),
            "p_v": float(vv.mean()),
            "bootstrap_sd_a11": bootstrap_sd_a11,
            "bootstrap_sd_energy": bootstrap_sd_energy,
            "bootstrap_repeats_valid": int(n_valid),
            "bootstrap_in_vector": False,
            "fit_scope": self.fit_scope,
            "patient_train_fingerprint": self.patient_train_fingerprint,
            "path": "ghcr_v1b_matched_aux",
        }

    def transform_pair(self, u_name: str, v_name: str) -> tuple[np.ndarray, dict]:
        assert self._train_df is not None
        self.required_pair_calls += 1
        u_s, v_s = str(u_name), str(v_name)

        # Fast zero path for V1/V1B non-binary (no cache pollution / expensive work).
        if self._is_v1() or self._is_v1b():
            if self._kind(u_s) != "binary" or self._kind(v_s) != "binary":
                vec = np.zeros(PAIR_DIM, dtype=np.float32)
                meta = {
                    "supported": False,
                    "kind_u": self._kind(u_s),
                    "kind_v": self._kind(v_s),
                    "path": "zero_nonbinary",
                    "cache_hit": False,
                    "fit_scope": self.fit_scope,
                    "patient_train_fingerprint": self.patient_train_fingerprint,
                }
                return vec, meta
            cache_key = self._v1_cache_key(u_s, v_s)
        else:
            cache_key = (str(self.config.variant), u_s, v_s, self.patient_train_fingerprint)

        if cache_key in self._pair_cache:
            self.cache_hits += 1
            vec_c, meta_c = self._pair_cache[cache_key]
            meta_out = dict(meta_c)
            meta_out["cache_hit"] = True
            meta_out["cache_key"] = repr(cache_key)
            return vec_c.copy(), meta_out

        self.cache_misses += 1
        vec, meta = self._transform_pair_uncached(u_s, v_s)
        meta = dict(meta)
        meta["cache_hit"] = False
        meta["cache_key"] = repr(cache_key)
        meta["fitted_on_n_patients"] = int(self.n_train_patients)
        meta["fit_scope"] = self.fit_scope
        meta["patient_train_fingerprint"] = self.patient_train_fingerprint
        self._pair_cache[cache_key] = (vec, meta)
        return vec.copy(), dict(meta)

    def _transform_pair_uncached(self, u_name: str, v_name: str) -> tuple[np.ndarray, dict]:
        assert self._train_df is not None
        col_u, col_v = self._column(u_name), self._column(v_name)
        if col_u not in self._train_df.columns or col_v not in self._train_df.columns:
            vec = np.zeros(PAIR_DIM, dtype=np.float32)
            return vec, {"supported": False, "reason": "missing_column"}

        u = pd.to_numeric(self._train_df[col_u], errors="coerce").to_numpy(dtype=float)
        v = pd.to_numeric(self._train_df[col_v], errors="coerce").to_numpy(dtype=float)
        complete = np.isfinite(u) & np.isfinite(v)
        n_complete = int(complete.sum())
        n_train = max(self.n_train_patients, int(len(u)), 1)
        # Completeness fraction — NOT joint-positive n11/N.
        support = float(n_complete / n_train)
        kind_u, kind_v = self._kind(u_name), self._kind(v_name)

        # --- V0 legacy binary_compact (only if encoder path used; V0 attach uses L2) ---
        if self._is_v0():
            if kind_u == "binary" and kind_v == "binary" and n_complete >= self.config.min_complete:
                vec = self._legacy_binary_compact(u, v)
                return vec, {
                    "supported": True,
                    "kind_u": kind_u,
                    "kind_v": kind_v,
                    "support": support,
                    "n_complete": n_complete,
                    "path": "legacy_binary_compact",
                }
            return np.zeros(PAIR_DIM, dtype=np.float32), {
                "supported": False,
                "kind_u": kind_u,
                "kind_v": kind_v,
                "support": support,
                "n_complete": n_complete,
                "path": "zero_nonbinary",
            }

        # --- V1: clean GHCR binary–binary only ---
        if self._is_v1():
            if not (kind_u == "binary" and kind_v == "binary"):
                return np.zeros(PAIR_DIM, dtype=np.float32), {
                    "supported": False,
                    "kind_u": kind_u,
                    "kind_v": kind_v,
                    "support": support,
                    "path": "zero_nonbinary",
                }
            return self._transform_v1_binary_pair(u, v, u_name=u_name, v_name=v_name)

        # --- V1B: GHCR a11 + matched V0 joint prevalence / rarity ---
        if self._is_v1b():
            if not (kind_u == "binary" and kind_v == "binary"):
                return np.zeros(PAIR_DIM, dtype=np.float32), {
                    "supported": False,
                    "kind_u": kind_u,
                    "kind_v": kind_v,
                    "path": "zero_nonbinary",
                }
            return self._transform_v1b_binary_pair(u, v, u_name=u_name, v_name=v_name)

        # --- V2: multi-type GHCR with globally fitted bases ---
        if n_complete < self.config.min_complete:
            return np.zeros(PAIR_DIM, dtype=np.float32), {
                "supported": False,
                "kind_u": kind_u,
                "kind_v": kind_v,
                "support": support,
                "n_complete": n_complete,
                "path": "insufficient_support",
            }

        if kind_u == "binary" and kind_v == "binary":
            n_u0, n_u1, n_v0, n_v1 = _binary_level_counts(u[complete], v[complete])
            min_level_count = int(min(n_u0, n_u1, n_v0, n_v1))
            if min_level_count < self.config.min_level_count:
                return np.zeros(PAIR_DIM, dtype=np.float32), {
                    "supported": False,
                    "kind_u": kind_u,
                    "kind_v": kind_v,
                    "support": support,
                    "n_complete": n_complete,
                    "min_level_count": min_level_count,
                    "path": "insufficient_level_count",
                }

        phi_u, _ = self._encode_series(u_name, u[complete])
        phi_v, _ = self._encode_series(v_name, v[complete])
        if phi_u.shape[1] == 0 or phi_v.shape[1] == 0:
            return np.zeros(PAIR_DIM, dtype=np.float32), {
                "supported": False,
                "kind_u": kind_u,
                "kind_v": kind_v,
                "path": "empty_basis",
            }

        matrix = self._coefficient_matrix(phi_u, phi_v)
        mapping = pair_kinds(kind_u, kind_v)

        rng = np.random.default_rng(self.config.bootstrap_seed + hash((u_name, v_name)) % 10_000)
        boots_a11: list[float] = []
        boots_energy: list[float] = []
        n = phi_u.shape[0]
        for _ in range(self.config.bootstrap_repeats):
            idx = rng.integers(0, n, size=n)
            m = self._coefficient_matrix(phi_u[idx], phi_v[idx])
            if m.size == 0:
                continue
            boots_energy.append(float(np.sum(m**2)))
            boots_a11.append(float(m[0, 0]) if m.shape[0] and m.shape[1] else 0.0)
        bootstrap_sd_a11 = float(np.std(boots_a11, ddof=1)) if len(boots_a11) > 1 else 0.0
        bootstrap_sd_energy = float(np.std(boots_energy, ddof=1)) if len(boots_energy) > 1 else 0.0
        # Vector slot: SD(a11) for binary–binary; else SD(energy) for multi-coeff pairs.
        bootstrap_sd = bootstrap_sd_a11 if mapping == "binary_binary" else bootstrap_sd_energy

        vec = pack_pair_vector(
            matrix,
            kind_u=kind_u,
            kind_v=kind_v,
            support=support,
            bootstrap_sd=bootstrap_sd,
            supported=True,
            binary_energy_is_a11_sq=(mapping == "binary_binary"),
        )
        stats = coefficient_stats(matrix, mapping)
        return vec, {
            "supported": True,
            "kind_u": kind_u,
            "kind_v": kind_v,
            "support": support,
            "n_complete": n_complete,
            "bootstrap_sd": bootstrap_sd,
            "bootstrap_sd_a11": bootstrap_sd_a11,
            "bootstrap_sd_energy": bootstrap_sd_energy,
            "path": "ghcr",
            **stats,
        }

    def transform_pairs(self, pairs: list[tuple[str, str]]) -> tuple[np.ndarray, list[dict]]:
        rows = []
        metas = []
        for u, v in pairs:
            vec, meta = self.transform_pair(str(u), str(v))
            rows.append(vec)
            metas.append(meta)
        if not rows:
            return np.zeros((0, PAIR_DIM), dtype=np.float32), []
        return np.stack(rows, axis=0).astype(np.float32), metas
