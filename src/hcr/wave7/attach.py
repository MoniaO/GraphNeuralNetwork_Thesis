"""Attach Wave 7 HCR features (pair 8-d → motif AZ⊕AG⊕ZG 24-d)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from data.candidate_pairs import candidate_pairs_from_data
from data.patient_matrix import load_patient_matrix_with_split, train_patient_df
from hcr.attach_l2_structural import fit_and_attach_l2_structural
from wnerw.wave5c.edge_context_registry import build_context_map
from wnerw.wave5c.runtime import ROOT as REPO_ROOT

from .encoder import Wave7HCRConfig, Wave7PairEncoder
from .mapping import PAIR_DIM
from .motif import (
    CONTEXT_SELECTION_RULE,
    MOTIF_TYPE,
    PAIR_ROLES,
    build_candidate_coparent_triples,
    triples_fingerprint,
)

MOTIF_DIM = 24


def wave7_enabled(cfg: Any) -> bool:
    wave = str(getattr(cfg.experiment, "wave", "")).upper()
    return "WAVE7" in wave or str(getattr(cfg.hcr, "encoder", "")).lower() == "generalized_hcr"


def _variant_name(cfg: Any) -> str:
    return str(
        getattr(cfg.experiment, "variant", None)
        or getattr(cfg.hcr, "variant", "W7_V0_L2_BINARY_COMPACT")
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


def _collect_required_pairs(
    splits,
    triples: dict[tuple[str, str], dict[str, Any]],
) -> list[tuple[str, str]]:
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


def fit_and_attach_wave7(
    cfg: Any,
    train_data,
    valid_data,
    test_data,
    device: torch.device | str = "cpu",
):
    """Fit Wave 7 HCR on train patients; attach 24-d motif features to splits."""
    variant = _variant_name(cfg)
    seed = int(getattr(cfg.data, "candidate_seed", getattr(cfg.training, "seed", 20260722)))

    # V0 reuses the audited L2 structural binary_compact attach (same triangle).
    if variant.upper().endswith("L2_BINARY_COMPACT") or variant.upper() == "W7_V0_L2_BINARY_COMPACT":
        enc = fit_and_attach_l2_structural(cfg, train_data, valid_data, test_data, device=device)
        for data in (train_data, valid_data, test_data):
            data.hcr_variant = "W7_V0_L2_BINARY_COMPACT"
            data.wave7_variant = "W7_V0_L2_BINARY_COMPACT"
            data.hcr_motif_type = MOTIF_TYPE
            data.hcr_pair_roles = list(PAIR_ROLES)
        return enc

    context_map, registry_hash, _registry_df = _load_context_map(seed)
    triples = build_candidate_coparent_triples(context_map)
    triples_hash = triples_fingerprint(triples)

    patient_df = load_patient_matrix_with_split(cfg)
    train_patients = train_patient_df(patient_df)
    scenario = str(getattr(cfg.data.dataset, "scenario", "clean"))

    hcr_cfg = Wave7HCRConfig.from_hydra(cfg.hcr, getattr(cfg, "experiment", None))
    hcr_cfg.variant = variant
    hcr_cfg.roles = PAIR_ROLES
    encoder = Wave7PairEncoder(hcr_cfg)
    encoder.scenario = scenario

    splits = (train_data, valid_data, test_data)
    required_pairs = _collect_required_pairs(splits, triples)
    unique_nodes = {n for uv in required_pairs for n in uv}
    encoder.fit(train_patients, unique_nodes)
    assert encoder.fit_scope == "patient_train_only"
    assert encoder.patient_train_fingerprint

    # Compute each unique pair once from patient_train; reuse across edge splits.
    pair_vectors: dict[tuple[str, str], np.ndarray] = {}
    pair_metas: dict[tuple[str, str], dict] = {}
    for u_name, v_name in required_pairs:
        vec, meta = encoder.transform_pair(u_name, v_name)
        pair_vectors[(u_name, v_name)] = vec
        pair_metas[(u_name, v_name)] = meta

    unique_pair_count = len(required_pairs)
    cache_hits = int(encoder.cache_hits)
    cache_misses = int(encoder.cache_misses)
    # Motif assembly below only does dict lookups — those are not transform_pair calls.
    # avoided ≈ (train+valid+test motif pair lookups) - unique computations
    motif_pair_lookups = 0
    for data in splits:
        for u, v in candidate_pairs_from_data(data):
            key = (str(u), str(v))
            motif_pair_lookups += 3 if key in triples else 1
    avoided = max(motif_pair_lookups - unique_pair_count, 0)

    out_dir = REPO_ROOT / "outputs" / "wave7" / "basis_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = out_dir / f"{variant}__{scenario}__seed{seed}.json"
    audit_payload = {
        "variant": variant,
        "motif_type": MOTIF_TYPE,
        "pair_roles": list(PAIR_ROLES),
        "context_selection_rule": CONTEXT_SELECTION_RULE,
        "context_registry_hash": registry_hash,
        "triples_hash": triples_hash,
        "n_triples": len(triples),
        "fit_scope": encoder.fit_scope,
        "patient_train_fingerprint": encoder.patient_train_fingerprint,
        "bootstrap_repeats": int(hcr_cfg.bootstrap_repeats),
        "required_pair_calls": int(encoder.required_pair_calls),
        "unique_pair_count": unique_pair_count,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "avoided_pair_computations": avoided,
        "motif_pair_lookups": motif_pair_lookups,
        "selector_note": (
            "Frozen Wave 5C registry. Dual/triple gates use expert_gate_schema parents "
            "from motif_registry_v3; other gates use G_train predecessors / "
            "select_structural_contexts. Not pure graph_train_parents for all edges."
        ),
        "basis_audit": encoder.basis_audit,
    }
    audit_path.write_text(json.dumps(audit_payload, indent=2, default=str))

    fp_path = REPO_ROOT / "outputs" / "wave7" / "MOTIF_FINGERPRINT.json"
    fp_path.write_text(
        json.dumps(
            {
                "motif_type": MOTIF_TYPE,
                "pair_roles": list(PAIR_ROLES),
                "context_selection_rule": CONTEXT_SELECTION_RULE,
                "context_registry_hash": registry_hash,
                "triples_hash": triples_hash,
                "seed": seed,
                "fit_scope": encoder.fit_scope,
                "patient_train_fingerprint": encoder.patient_train_fingerprint,
            },
            indent=2,
        )
    )

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
                    **{f"hcr_{j}": float(matrix[i, j]) for j in range(MOTIF_DIM)},
                }
            )

        data.hcr_features = torch.as_tensor(matrix, dtype=torch.float32, device=device)
        data.hcr_supported = torch.tensor(supported, dtype=torch.bool, device=device)
        data.hcr_motif_available = torch.tensor(motif_available, dtype=torch.bool, device=device)
        data.hcr_enabled = True
        data.hcr_variant = variant
        data.wave7_variant = variant
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

    cache_dir = REPO_ROOT / "outputs" / "hcr" / "wave7"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{variant}__{scenario}.csv"
    pd.DataFrame(cache_rows).drop_duplicates(subset=["source", "target"]).to_csv(
        cache_path, index=False
    )
    print(
        f"\nHCR WAVE7\n  variant: {variant}\n  motif: {MOTIF_TYPE} {PAIR_ROLES}\n"
        f"  dim: {MOTIF_DIM}\n  contexts: {len(triples)}\n"
        f"  fit_scope: {encoder.fit_scope}\n"
        f"  patient_train_fingerprint: {encoder.patient_train_fingerprint[:16]}…\n"
        f"  unique_pairs: {unique_pair_count}  cache_hits: {cache_hits}  "
        f"cache_misses: {cache_misses}  avoided: {avoided}\n"
        f"  registry_hash: {registry_hash[:12]}…\n  triples_hash: {triples_hash[:12]}…\n"
        f"  audit: {audit_path}\n  cache: {cache_path}"
    )
    return encoder
