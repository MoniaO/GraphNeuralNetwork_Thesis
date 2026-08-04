"""Attach Wave 7 Panel B HCR features (pair 40-d → motif AZ⊕AG⊕ZG 120-d)."""

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
from hcr.attach_l2_structural import fit_and_attach_l2_structural
from hcr.wave7.motif import (
    CONTEXT_SELECTION_RULE,
    MOTIF_TYPE,
    PAIR_ROLES,
    build_candidate_coparent_triples,
    triples_fingerprint,
)
from wnerw.wave5c.edge_context_registry import build_context_map
from wnerw.wave5c.runtime import ROOT as REPO_ROOT

from .encoder import PanelBConfig, PanelBPairEncoder
from .packing import PAIR_DIM

MOTIF_DIM = 120


def panel_b_enabled(cfg: Any) -> bool:
    wave = str(getattr(cfg.experiment, "wave", "")).upper()
    variant = str(
        getattr(cfg.experiment, "variant", None) or getattr(cfg.hcr, "variant", "")
    ).upper()
    # Residual mini-wave has its own attach path.
    if variant.startswith("W7BR_") or "PANEL_B_RESIDUAL" in wave:
        return False
    if "W7B_" in variant or ( "PANEL_B" in wave and "RESIDUAL" not in wave):
        return True
    if "WAVE7" in wave and ("JITTER" in variant or "PAD40" in variant or "ENRICHED40" in variant):
        return True
    return str(getattr(cfg.hcr, "encoder", "")).lower() == "panel_b_jitter_ghcr"


def _variant_name(cfg: Any) -> str:
    return str(
        getattr(cfg.experiment, "variant", None)
        or getattr(cfg.hcr, "variant", "W7B_B3_HYBRID_LEGACY_BINARY_JITTER40")
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


def _load_context_map(seed: int) -> tuple[dict[tuple[str, str], tuple[str, ...]], str, pd.DataFrame]:
    reg = _load_registry(seed)
    return build_context_map(reg), _registry_hash(reg), reg


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


def _pad_v0_motif_to_120(motif24: np.ndarray) -> np.ndarray:
    """Expand each legacy 8-d pair block into a 40-d slot (zeros in 8..39)."""
    m = np.asarray(motif24, dtype=np.float32)
    if m.ndim == 1:
        m = m.reshape(1, -1)
    n = m.shape[0]
    out = np.zeros((n, MOTIF_DIM), dtype=np.float32)
    for b in range(3):
        src = m[:, b * 8 : (b + 1) * 8]
        out[:, b * PAIR_DIM : b * PAIR_DIM + 8] = src
    return out


def _attach_b0_legacy_pad(
    cfg: Any,
    train_data,
    valid_data,
    test_data,
    device: torch.device | str,
    variant: str,
):
    """Width-control: exact Wave 7 V0 values, zero-padded to 120-d."""
    t0 = time.perf_counter()
    enc = fit_and_attach_l2_structural(cfg, train_data, valid_data, test_data, device=device)
    attach_s = time.perf_counter() - t0
    for data in (train_data, valid_data, test_data):
        old = data.hcr_features.detach().cpu().numpy()
        assert old.shape[1] == 24, f"expected V0 motif 24-d, got {old.shape}"
        padded = _pad_v0_motif_to_120(old)
        data.hcr_features = torch.as_tensor(padded, dtype=torch.float32, device=device)
        data.hcr_dim = MOTIF_DIM
        data.hcr_variant = variant
        data.wave7_variant = variant
        data.wave7b_variant = variant
        data.hcr_motif_type = MOTIF_TYPE
        data.hcr_pair_roles = list(PAIR_ROLES)
        data.hcr_panel = "B"
        data.hcr_attach_seconds = float(attach_s)
    print(
        f"\nHCR WAVE7 PANEL B (B0 pad)\n  variant: {variant}\n"
        f"  motif: {MOTIF_TYPE} {PAIR_ROLES}\n  dim: {MOTIF_DIM}\n"
        f"  source: exact L2/V0 binary_compact padded 8→40 per role\n"
        f"  attach_s: {attach_s:.2f}"
    )
    return enc


def fit_and_attach_panel_b(
    cfg: Any,
    train_data,
    valid_data,
    test_data,
    device: torch.device | str = "cpu",
):
    """Fit Panel B on train patients; attach 120-d motif features to splits."""
    variant = _variant_name(cfg)
    seed = int(getattr(cfg.data, "candidate_seed", getattr(cfg.training, "seed", 20260722)))
    mode = PanelBConfig(variant=variant).mode()

    if mode == "B0":
        return _attach_b0_legacy_pad(cfg, train_data, valid_data, test_data, device, variant)

    t0 = time.perf_counter()
    context_map, registry_hash, _reg = _load_context_map(seed)
    triples = build_candidate_coparent_triples(context_map)
    triples_hash = triples_fingerprint(triples)

    patient_df = load_patient_matrix_with_split(cfg)
    train_patients = train_patient_df(patient_df)
    scenario = str(getattr(cfg.data.dataset, "scenario", "clean"))

    hcr_cfg = PanelBConfig.from_hydra(cfg.hcr, getattr(cfg, "experiment", None))
    hcr_cfg.variant = variant
    encoder = PanelBPairEncoder(hcr_cfg)
    encoder.scenario = scenario

    splits = (train_data, valid_data, test_data)
    required_pairs = _collect_required_pairs(splits, triples)
    unique_nodes = {n for uv in required_pairs for n in uv}
    encoder.fit(train_patients, unique_nodes)
    assert encoder.fit_scope == "patient_train_only"
    assert encoder.patient_train_fingerprint

    pair_vectors: dict[tuple[str, str], np.ndarray] = {}
    pair_metas: dict[tuple[str, str], dict] = {}
    for u_name, v_name in required_pairs:
        vec, meta = encoder.transform_pair(u_name, v_name)
        assert vec.shape == (PAIR_DIM,), f"pair dim {vec.shape} != {PAIR_DIM}"
        assert np.isfinite(vec).all(), f"NaN/Inf in pair ({u_name},{v_name})"
        pair_vectors[(u_name, v_name)] = vec
        pair_metas[(u_name, v_name)] = meta

    unique_pair_count = len(required_pairs)
    motif_pair_lookups = 0
    for data in splits:
        for u, v in candidate_pairs_from_data(data):
            key = (str(u), str(v))
            motif_pair_lookups += 3 if key in triples else 1
    avoided = max(motif_pair_lookups - unique_pair_count, 0)

    out_dir = REPO_ROOT / "outputs" / "wave7" / "panel_b" / "basis_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = out_dir / f"{variant}__{scenario}__seed{seed}.json"
    audit_payload = {
        "panel": "B",
        "variant": variant,
        "mode": mode,
        "motif_type": MOTIF_TYPE,
        "pair_roles": list(PAIR_ROLES),
        "pair_dim": PAIR_DIM,
        "motif_dim": MOTIF_DIM,
        "context_selection_rule": CONTEXT_SELECTION_RULE,
        "context_registry_hash": registry_hash,
        "triples_hash": triples_hash,
        "n_triples": len(triples),
        "fit_scope": encoder.fit_scope,
        "patient_train_fingerprint": encoder.patient_train_fingerprint,
        "bootstrap_in_vector": False,
        "unique_pair_count": unique_pair_count,
        "cache_hits": int(encoder.cache_hits),
        "cache_misses": int(encoder.cache_misses),
        "avoided_pair_computations": avoided,
        "motif_pair_lookups": motif_pair_lookups,
    }
    audit_path.write_text(json.dumps(audit_payload, indent=2, default=str))

    cache_rows = []
    for data in splits:
        pairs = candidate_pairs_from_data(data)
        matrix = np.zeros((len(pairs), MOTIF_DIM), dtype=np.float32)
        supported = []
        motif_available = []
        for i, (u, v) in enumerate(pairs):
            key = (str(u), str(v))
            if key in triples:
                meta_t = triples[key]
                parts = []
                metas = []
                for p in meta_t["pairs"]:
                    pk = (str(p[0]), str(p[1]))
                    parts.append(pair_vectors[pk])
                    metas.append(pair_metas[pk])
                matrix[i] = np.concatenate(parts).astype(np.float32)
                supported.append(all(m.get("supported", False) for m in metas))
                motif_available.append(True)
                selected_context = meta_t["selected_context"]
                context_candidates = meta_t["context_candidates"]
            else:
                vec = pair_vectors[key]
                meta = pair_metas[key]
                matrix[i, :PAIR_DIM] = vec
                supported.append(bool(meta.get("supported", False)))
                motif_available.append(False)
                selected_context = ""
                context_candidates = []

            cache_rows.append(
                {
                    "source": key[0],
                    "target": key[1],
                    "motif_type": MOTIF_TYPE,
                    "motif_available": motif_available[-1],
                    "supported_motif": supported[-1],
                    "selected_context": selected_context,
                    "context_candidates": json.dumps(context_candidates),
                    "context_selection_rule": CONTEXT_SELECTION_RULE,
                    "pair_roles": json.dumps(list(PAIR_ROLES)),
                    "fit_scope": encoder.fit_scope,
                    "patient_train_fingerprint": encoder.patient_train_fingerprint,
                    **{f"hcr_{j}": float(matrix[i, j]) for j in range(min(MOTIF_DIM, 24))},
                }
            )

        assert matrix.shape[1] == MOTIF_DIM
        assert np.isfinite(matrix).all()
        data.hcr_features = torch.as_tensor(matrix, dtype=torch.float32, device=device)
        data.hcr_supported = torch.tensor(supported, dtype=torch.bool, device=device)
        data.hcr_motif_available = torch.tensor(motif_available, dtype=torch.bool, device=device)
        data.hcr_enabled = True
        data.hcr_variant = variant
        data.wave7_variant = variant
        data.wave7b_variant = variant
        data.hcr_dim = MOTIF_DIM
        data.hcr_motif_type = MOTIF_TYPE
        data.hcr_pair_roles = list(PAIR_ROLES)
        data.hcr_fit_split = "train"
        data.hcr_fit_scope = encoder.fit_scope
        data.hcr_n_train_patients = int(len(train_patients))
        data.hcr_patient_train_fingerprint = encoder.patient_train_fingerprint
        data.hcr_basis_audit_path = str(audit_path)
        data.hcr_context_registry_hash = registry_hash
        data.hcr_triples_hash = triples_hash
        data.hcr_panel = "B"

    attach_s = time.perf_counter() - t0
    for data in splits:
        data.hcr_attach_seconds = float(attach_s)

    cache_dir = REPO_ROOT / "outputs" / "hcr" / "wave7_panel_b"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{variant}__{scenario}.csv"
    # Store compact metadata; full 120 floats would make a huge CSV.
    meta_cols = [
        c
        for c in cache_rows[0].keys()
        if not str(c).startswith("hcr_") or int(str(c).split("_")[1]) < 8
    ] if cache_rows else []
    pd.DataFrame([{k: r[k] for k in meta_cols} for r in cache_rows]).drop_duplicates(
        subset=["source", "target"]
    ).to_csv(cache_path, index=False)

    print(
        f"\nHCR WAVE7 PANEL B\n  variant: {variant}\n  mode: {mode}\n"
        f"  motif: {MOTIF_TYPE} {PAIR_ROLES}\n  dim: {MOTIF_DIM}\n"
        f"  contexts: {len(triples)}\n  fit_scope: {encoder.fit_scope}\n"
        f"  patient_train_fingerprint: {encoder.patient_train_fingerprint[:16]}…\n"
        f"  unique_pairs: {unique_pair_count}  cache_hits: {encoder.cache_hits}  "
        f"cache_misses: {encoder.cache_misses}  avoided: {avoided}\n"
        f"  registry_hash: {registry_hash[:12]}…\n  triples_hash: {triples_hash[:12]}…\n"
        f"  attach_s: {attach_s:.2f}\n  audit: {audit_path}\n  cache: {cache_path}"
    )
    return encoder
