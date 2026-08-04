"""Attach dual motifs for residual fusion: B2 120-d + V0 24-d + BB masks."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from data.candidate_pairs import candidate_pairs_from_data
from data.patient_matrix import load_patient_matrix_with_split, train_patient_df
from hcr.pair_encoder import HCRPairEncoder
from hcr.variable_spec import VariableType
from hcr.variable_specs_v3 import VARIABLE_SPECS
from hcr.wave7.motif import (
    CONTEXT_SELECTION_RULE,
    MOTIF_TYPE,
    PAIR_ROLES,
    build_candidate_coparent_triples,
    triples_fingerprint,
)
from hcr.wave7.panel_b.encoder import PanelBConfig, PanelBPairEncoder
from hcr.wave7.panel_b.packing import PAIR_DIM, legacy_binary_compact_8
from wnerw.wave5c.edge_context_registry import build_context_map
from wnerw.wave5c.runtime import ROOT as REPO_ROOT

from .conditional_edge_gain import TRIPLE_DIM, compute_conditional_edge_gain
from .constants import (
    CLASSICAL4_V0_SLOTS,
    MOTIF_B2_DIM,
    MOTIF_CLASSICAL4_DIM,
    MOTIF_V0_DIM,
    N_ROLES,
    PAIR_CLASSICAL4_DIM,
    PAIR_V0_DIM,
)
from .correlation_audit import write_v0_vs_b2_correlations

B2_VARIANT = "W7B_B2_JITTER_GHCR_ENRICHED40"


def residual_enabled(cfg: Any) -> bool:
    wave = str(getattr(cfg.experiment, "wave", "")).upper()
    variant = str(
        getattr(cfg.experiment, "variant", None) or getattr(cfg.hcr, "variant", "")
    ).upper()
    if variant.startswith("W7BR_") or "PANEL_B_RESIDUAL" in wave:
        return True
    return str(getattr(cfg.hcr, "encoder", "")).lower() == "panel_b_residual_fusion"


def _variant_name(cfg: Any) -> str:
    return str(
        getattr(cfg.experiment, "variant", None)
        or getattr(cfg.hcr, "variant", "W7BR_R2_B2_GATED_LEGACY_RESIDUAL")
    ).strip()


def _load_registry(seed: int) -> pd.DataFrame:
    reg_path = REPO_ROOT / "outputs/wave5c/registry/edge_context_registry.csv"
    if not reg_path.exists():
        return pd.DataFrame()
    reg = pd.read_csv(reg_path)
    if "seed" in reg.columns:
        reg = reg[reg["seed"] == int(seed)]
    return reg


def _registry_hash(reg: pd.DataFrame) -> str:
    if reg is None or len(reg) == 0:
        return hashlib.sha256(b"empty").hexdigest()
    cols = [c for c in ("source", "target", "context_nodes", "selector_source") if c in reg.columns]
    blob = reg.loc[:, cols].sort_values(cols).to_csv(index=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _collect_required_pairs(splits, triples) -> list[tuple[str, str]]:
    required: set[tuple[str, str]] = set()
    for data in splits:
        for u, v in candidate_pairs_from_data(data):
            key = (str(u), str(v))
            if key in triples:
                for p in triples[key]["pairs"]:
                    required.add((str(p[0]), str(p[1])))
            else:
                required.add(key)
    return sorted(required)


def _kind(name: str) -> str:
    spec = VARIABLE_SPECS.get(str(name))
    if spec is None:
        return "unknown"
    if spec.variable_type == VariableType.BINARY:
        return "binary"
    if spec.variable_type == VariableType.COUNT:
        return "count"
    return "continuous"


def _is_binary_binary(u: str, v: str) -> bool:
    return _kind(u) == "binary" and _kind(v) == "binary"


def _is_binary_triple(a: str, z: str, g: str) -> bool:
    return _kind(a) == "binary" and _kind(z) == "binary" and _kind(g) == "binary"


def _column_values(train_df: pd.DataFrame, name: str) -> np.ndarray | None:
    spec = VARIABLE_SPECS.get(str(name))
    col = spec.column_name if spec is not None else str(name)
    if col not in train_df.columns:
        return None
    return pd.to_numeric(train_df[col], errors="coerce").to_numpy(dtype=float)


def _v0_to_classical4(v0: np.ndarray) -> np.ndarray:
    v = np.asarray(v0, dtype=np.float32).reshape(-1)
    if v.shape[0] < PAIR_V0_DIM:
        out = np.zeros(PAIR_CLASSICAL4_DIM, dtype=np.float32)
        return out
    return v[list(CLASSICAL4_V0_SLOTS)].astype(np.float32)


class _HCR2Cfg:
    variant = "binary_compact"
    smoothing = 0.5
    output_dim = 8
    unknown_pair_value = 0.0
    unsupported_pair_mode = "zeros"
    force_zero_features = False
    append_supported_mask = False
    features = None


def _legacy_v0_pair(
    enc: HCRPairEncoder,
    u: str,
    v: str,
    train_df: pd.DataFrame,
) -> np.ndarray:
    """Exact binary_compact 8-d via HCRPairEncoder; zero if non-binary or missing."""
    if not _is_binary_binary(u, v):
        return np.zeros(PAIR_V0_DIM, dtype=np.float32)
    try:
        vec = enc.transform([(str(u), str(v))]).cpu().numpy()[0]
        return np.asarray(vec, dtype=np.float32).reshape(-1)[:PAIR_V0_DIM]
    except Exception:
        # Fallback: direct computation from train columns (same function as Panel B).
        spec_u = VARIABLE_SPECS.get(str(u))
        spec_v = VARIABLE_SPECS.get(str(v))
        if spec_u is None or spec_v is None:
            return np.zeros(PAIR_V0_DIM, dtype=np.float32)
        col_u, col_v = spec_u.column_name, spec_v.column_name
        if col_u not in train_df.columns or col_v not in train_df.columns:
            return np.zeros(PAIR_V0_DIM, dtype=np.float32)
        uu = pd.to_numeric(train_df[col_u], errors="coerce").to_numpy(dtype=float)
        vv = pd.to_numeric(train_df[col_v], errors="coerce").to_numpy(dtype=float)
        return legacy_binary_compact_8(uu, vv, smoothing=0.5)


def fit_and_attach_panel_b_residual(
    cfg: Any,
    train_data,
    valid_data,
    test_data,
    device: torch.device | str = "cpu",
):
    """Attach B2 120-d + V0 24-d + per-role binary–binary masks. Features train-only."""
    variant = _variant_name(cfg)
    seed = int(getattr(cfg.data, "candidate_seed", getattr(cfg.training, "seed", 20260722)))
    t0 = time.perf_counter()

    reg = _load_registry(seed)
    context_map = build_context_map(reg)
    registry_hash = _registry_hash(reg)
    triples = build_candidate_coparent_triples(context_map)
    triples_hash = triples_fingerprint(triples)

    patient_df = load_patient_matrix_with_split(cfg)
    train_patients = train_patient_df(patient_df)
    scenario = str(getattr(cfg.data.dataset, "scenario", "clean"))

    # --- B2 encoder (exact Panel B B2 config; do not alter packing) ---
    b2_cfg = PanelBConfig.from_hydra(cfg.hcr, getattr(cfg, "experiment", None))
    b2_cfg.variant = B2_VARIANT
    b2_encoder = PanelBPairEncoder(b2_cfg)
    b2_encoder.scenario = scenario

    splits = (train_data, valid_data, test_data)
    required_pairs = _collect_required_pairs(splits, triples)
    unique_nodes = {n for uv in required_pairs for n in uv}
    b2_encoder.fit(train_patients, unique_nodes)
    assert b2_encoder.fit_scope == "patient_train_only"

    # --- V0 / L2 binary_compact encoder (same as Wave 7 V0) ---
    v0_encoder = HCRPairEncoder.from_hydra(_HCR2Cfg(), VARIABLE_SPECS)
    v0_encoder.fit(train_patient_df=train_patients, candidate_pairs=required_pairs)

    b2_pairs: dict[tuple[str, str], np.ndarray] = {}
    v0_pairs: dict[tuple[str, str], np.ndarray] = {}
    bb_flags: dict[tuple[str, str], bool] = {}
    for u_name, v_name in required_pairs:
        key = (str(u_name), str(v_name))
        b2_vec, _meta = b2_encoder.transform_pair(u_name, v_name)
        assert b2_vec.shape == (PAIR_DIM,)
        assert np.isfinite(b2_vec).all()
        b2_pairs[key] = b2_vec
        v0_pairs[key] = _legacy_v0_pair(v0_encoder, u_name, v_name, train_patients)
        assert v0_pairs[key].shape == (PAIR_V0_DIM,)
        bb_flags[key] = _is_binary_binary(u_name, v_name)
        # Nonbinary legacy must be exactly zero.
        if not bb_flags[key]:
            assert np.allclose(v0_pairs[key], 0.0), f"legacy leak for non-BB {key}"

    # Correlation audit on unique BB pairs (train-fit features only).
    bb_keys = [k for k, is_bb in bb_flags.items() if is_bb]
    audit_dir = REPO_ROOT / "outputs" / "wave7" / "panel_b_residual" / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / "V0_VS_B2_BINARY_CORRELATIONS.csv"
    if bb_keys:
        v0_mat = np.stack([v0_pairs[k] for k in bb_keys], axis=0)
        b2_mat = np.stack([b2_pairs[k] for k in bb_keys], axis=0)
        write_v0_vs_b2_correlations(v0_vectors=v0_mat, b2_vectors=b2_mat, out_path=audit_path)
    else:
        pd.DataFrame().to_csv(audit_path, index=False)

    # --- Triplewise conditional edge gain for each motif (A,Z,G); ordered cache ---
    n_train = int(len(train_patients))
    triple_cache: dict[tuple[str, str, str], object] = {}
    audit_rows: list[dict] = []
    for (a_name, g_name), meta_t in triples.items():
        z_name = str(meta_t["Z"])
        key = (str(a_name), str(z_name), str(g_name))  # directed; NOT sorted
        if key in triple_cache:
            continue
        if not _is_binary_triple(a_name, z_name, g_name):
            from .conditional_edge_gain import ConditionalEdgeGainResult

            res = ConditionalEdgeGainResult(
                vector=np.zeros(TRIPLE_DIM, dtype=np.float32),
                n_raw=np.zeros((2, 2, 2), dtype=np.int64),
                n_complete=0,
                h_g_given_z=0.0,
                h_g_given_az=0.0,
                ig=0.0,
                ig_norm=0.0,
                rd_cond=0.0,
                rd_z0=0.0,
                rd_z1=0.0,
                interaction=0.0,
                support=0.0,
                mask=0.0,
                supported=False,
                reason="non_binary_triple",
            )
        else:
            av = _column_values(train_patients, a_name)
            zv = _column_values(train_patients, z_name)
            gv = _column_values(train_patients, g_name)
            if av is None or zv is None or gv is None:
                res = compute_conditional_edge_gain(
                    np.array([]), np.array([]), np.array([]), n_train=n_train
                )
            else:
                res = compute_conditional_edge_gain(av, zv, gv, n_train=n_train)
        triple_cache[key] = res
        audit_rows.append(
            {
                "A": key[0],
                "Z": key[1],
                "G": key[2],
                "motif_role": "AZG",
                "n000": int(res.n_raw[0, 0, 0]),
                "n001": int(res.n_raw[0, 0, 1]),
                "n010": int(res.n_raw[0, 1, 0]),
                "n011": int(res.n_raw[0, 1, 1]),
                "n100": int(res.n_raw[1, 0, 0]),
                "n101": int(res.n_raw[1, 0, 1]),
                "n110": int(res.n_raw[1, 1, 0]),
                "n111": int(res.n_raw[1, 1, 1]),
                "n_complete": res.n_complete,
                "H_G_given_Z": res.h_g_given_z,
                "H_G_given_AZ": res.h_g_given_az,
                "conditional_edge_gain": res.ig,
                "normalized_edge_gain": res.ig_norm,
                "conditional_RD": res.rd_cond,
                "RD_Z0": res.rd_z0,
                "RD_Z1": res.rd_z1,
                "interaction_heterogeneity": res.interaction,
                "support": res.support,
                "triple_supported_mask": res.mask,
                "reason": res.reason,
            }
        )
    cmi_audit_path = audit_dir / "CONDITIONAL_EDGE_GAIN_BINARY_TRIPLES.csv"
    pd.DataFrame(audit_rows).to_csv(cmi_audit_path, index=False)

    basis_dir = REPO_ROOT / "outputs" / "wave7" / "panel_b_residual" / "basis_audit"
    basis_dir.mkdir(parents=True, exist_ok=True)
    basis_path = basis_dir / f"{variant}__{scenario}__seed{seed}.json"
    basis_path.write_text(
        json.dumps(
            {
                "panel": "B_RESIDUAL",
                "variant": variant,
                "b2_variant_source": B2_VARIANT,
                "motif_type": MOTIF_TYPE,
                "pair_roles": list(PAIR_ROLES),
                "b2_motif_dim": MOTIF_B2_DIM,
                "v0_motif_dim": MOTIF_V0_DIM,
                "classical4_motif_dim": MOTIF_CLASSICAL4_DIM,
                "triple_dim": TRIPLE_DIM,
                "fit_scope": b2_encoder.fit_scope,
                "patient_train_fingerprint": b2_encoder.patient_train_fingerprint,
                "context_registry_hash": registry_hash,
                "triples_hash": triples_hash,
                "n_required_pairs": len(required_pairs),
                "n_binary_binary_pairs": len(bb_keys),
                "n_triple_cache": len(triple_cache),
                "n_triple_supported": int(
                    sum(1 for r in triple_cache.values() if getattr(r, "supported", False))
                ),
                "correlation_audit": str(audit_path),
                "conditional_edge_gain_audit": str(cmi_audit_path),
            },
            indent=2,
        )
    )

    for data in splits:
        pairs = candidate_pairs_from_data(data)
        b2_matrix = np.zeros((len(pairs), MOTIF_B2_DIM), dtype=np.float32)
        v0_matrix = np.zeros((len(pairs), MOTIF_V0_DIM), dtype=np.float32)
        c4_matrix = np.zeros((len(pairs), MOTIF_CLASSICAL4_DIM), dtype=np.float32)
        triple_matrix = np.zeros((len(pairs), TRIPLE_DIM), dtype=np.float32)
        bb_mask = np.zeros((len(pairs), N_ROLES), dtype=np.float32)
        supported = []
        motif_available = []

        for i, (u, v) in enumerate(pairs):
            key = (str(u), str(v))
            if key in triples:
                meta_t = triples[key]
                b2_parts = []
                v0_parts = []
                c4_parts = []
                for r, p in enumerate(meta_t["pairs"]):
                    pk = (str(p[0]), str(p[1]))
                    b2_parts.append(b2_pairs[pk])
                    v0_parts.append(v0_pairs[pk])
                    c4_parts.append(_v0_to_classical4(v0_pairs[pk]))
                    bb_mask[i, r] = 1.0 if bb_flags[pk] else 0.0
                b2_matrix[i] = np.concatenate(b2_parts)
                v0_matrix[i] = np.concatenate(v0_parts)
                c4_matrix[i] = np.concatenate(c4_parts)
                tkey = (str(meta_t["A"]), str(meta_t["Z"]), str(meta_t["G"]))
                tres = triple_cache[tkey]
                triple_matrix[i] = np.asarray(tres.vector, dtype=np.float32)
                supported.append(True)
                motif_available.append(True)
            else:
                b2_matrix[i, :PAIR_DIM] = b2_pairs[key]
                v0_matrix[i, :PAIR_V0_DIM] = v0_pairs[key]
                c4_matrix[i, :PAIR_CLASSICAL4_DIM] = _v0_to_classical4(v0_pairs[key])
                bb_mask[i, 0] = 1.0 if bb_flags[key] else 0.0
                # No context Z → triple block stays zero.
                supported.append(True)
                motif_available.append(False)

        assert b2_matrix.shape[1] == MOTIF_B2_DIM
        assert v0_matrix.shape[1] == MOTIF_V0_DIM
        assert c4_matrix.shape[1] == MOTIF_CLASSICAL4_DIM
        assert triple_matrix.shape[1] == TRIPLE_DIM
        assert np.isfinite(b2_matrix).all() and np.isfinite(v0_matrix).all()
        assert np.isfinite(c4_matrix).all() and np.isfinite(triple_matrix).all()
        for r in range(N_ROLES):
            inactive = bb_mask[:, r] < 0.5
            block = v0_matrix[:, r * PAIR_V0_DIM : (r + 1) * PAIR_V0_DIM]
            assert np.allclose(block[inactive], 0.0)
            cblock = c4_matrix[:, r * PAIR_CLASSICAL4_DIM : (r + 1) * PAIR_CLASSICAL4_DIM]
            assert np.allclose(cblock[inactive], 0.0)

        data.hcr_features = torch.as_tensor(b2_matrix, dtype=torch.float32, device=device)
        data.hcr_legacy_features = torch.as_tensor(
            v0_matrix, dtype=torch.float32, device=device
        )
        data.hcr_classical4_features = torch.as_tensor(
            c4_matrix, dtype=torch.float32, device=device
        )
        data.hcr_triple_features = torch.as_tensor(
            triple_matrix, dtype=torch.float32, device=device
        )
        data.hcr_bb_mask = torch.as_tensor(bb_mask, dtype=torch.float32, device=device)
        data.hcr_supported = torch.tensor(supported, dtype=torch.bool, device=device)
        data.hcr_motif_available = torch.tensor(motif_available, dtype=torch.bool, device=device)
        data.hcr_enabled = True
        data.hcr_variant = variant
        data.wave7_variant = variant
        data.hcr_dim = MOTIF_B2_DIM
        data.hcr_legacy_dim = MOTIF_V0_DIM
        data.hcr_classical4_dim = MOTIF_CLASSICAL4_DIM
        data.hcr_triple_dim = TRIPLE_DIM
        data.hcr_motif_type = MOTIF_TYPE
        data.hcr_pair_roles = list(PAIR_ROLES)
        data.hcr_fit_split = "train"
        data.hcr_fit_scope = b2_encoder.fit_scope
        data.hcr_n_train_patients = int(len(train_patients))
        data.hcr_patient_train_fingerprint = b2_encoder.patient_train_fingerprint
        data.hcr_basis_audit_path = str(basis_path)
        data.hcr_context_registry_hash = registry_hash
        data.hcr_triples_hash = triples_hash
        data.hcr_panel = "B_RESIDUAL"
        data.hcr_correlation_audit_path = str(audit_path)
        data.hcr_cmi_audit_path = str(cmi_audit_path)

    attach_s = time.perf_counter() - t0
    for data in splits:
        data.hcr_attach_seconds = float(attach_s)

    n_trip_sup = int(sum(1 for r in triple_cache.values() if getattr(r, "supported", False)))
    print(
        f"\nHCR WAVE7 PANEL B RESIDUAL\n  variant: {variant}\n"
        f"  B2 source: {B2_VARIANT} → {MOTIF_B2_DIM}D\n"
        f"  V0 legacy: binary_compact → {MOTIF_V0_DIM}D\n"
        f"  classical4: → {MOTIF_CLASSICAL4_DIM}D\n"
        f"  triple CMI: → {TRIPLE_DIM}D (supported {n_trip_sup}/{len(triple_cache)})\n"
        f"  motif: {MOTIF_TYPE} {PAIR_ROLES}\n"
        f"  fit_scope: {b2_encoder.fit_scope}\n"
        f"  patient_train_fingerprint: {b2_encoder.patient_train_fingerprint[:16]}…\n"
        f"  registry_hash: {registry_hash[:12]}…\n"
        f"  triples_hash: {triples_hash[:12]}…\n"
        f"  BB pairs audited: {len(bb_keys)}\n"
        f"  correlation_audit: {audit_path}\n"
        f"  cmi_audit: {cmi_audit_path}\n"
        f"  attach_s: {attach_s:.2f}"
    )
    return b2_encoder
