"""Attach S0–S10 features to candidate edges (train-only fit).

What it does
------------
1. Computes bases / NMI / Jaccard / FULL40 on train patients only.
2. For each candidate pair builds stat_raw [N, 3, D] (roles AZ, AG, ZG)
   and role masks (missing Z → mask 0, no imputation).
3. Writes this onto train/valid/test HeteroData — same fit, zero leakage.

What you may change
-------------------
- variant via `experiment.stat_variant` (S0–S10). FINAL = S10_HCR_FULL40.
- nothing else if you reproduce 14.08.

What not to touch for FINAL
---------------------------
Fit on train only. Z comes from the context registry
(`outputs/taskA/context_registry/`), never from G_true or edge labels.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from taskA.data.candidate_pairs import candidate_pairs_from_data
from taskA.data.patient_matrix import load_patient_matrix_with_split, train_patient_df
from taskA.features.pair_basis.hcr40.motif import (
    PAIR_ROLES,
    build_candidate_coparent_triples,
    triples_fingerprint,
)
from taskA.features.context.coparent.edge_context_registry import build_context_map
from taskA.features.context.coparent.runtime import ROOT as REPO_ROOT

from .compute import StageCFeatureStore, fit_stage_c_features
from .variants import StageCVariant, get_variant

N_ROLES = 3  # AZ, AG, ZG


def stage_c_enabled(cfg: Any) -> bool:
    exp = getattr(cfg, "experiment", None)
    if exp is None:
        return False
    wave = str(getattr(exp, "wave", "")).upper()
    variant = str(getattr(exp, "variant", "") or "").upper()
    stat = str(getattr(exp, "stat_variant", "") or "").upper()
    if stat:
        return True
    if "STAGE_C" in variant or "STAGE_C" in wave:
        return True
    if "FINAL_LARGE_GRID" in wave and "STAGE_C" in variant:
        return True
    return False


def _load_registry(seed: int):
    candidates = [
        REPO_ROOT / "outputs/taskA/context_registry/edge_context_registry.csv",
        REPO_ROOT / "outputs/wave5c/registry/edge_context_registry.csv",
    ]
    path = next((p for p in candidates if p.exists()), candidates[0])
    if not path.exists():
        raise FileNotFoundError(
            f"Missing coparent edge-context registry: {path}. "
            "Build with scripts/taskA/00_build_context_registry.py"
        )
    import pandas as pd

    reg = pd.read_csv(path)
    if "seed" in reg.columns:
        reg = reg[reg["seed"] == int(seed)]
    if len(reg) == 0:
        raise ValueError(f"Registry has no rows for seed={seed}")
    return reg


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


def _resolve_variant(cfg: Any) -> StageCVariant:
    exp = getattr(cfg, "experiment", None)
    raw = (
        getattr(exp, "stat_variant", None)
        or getattr(exp, "variant", None)
        or getattr(getattr(cfg, "hcr", None), "variant", None)
        or "S0_NONE"
    )
    s = str(raw)
    # Strip STAGE_C_ prefix if present
    if s.upper().startswith("STAGE_C_"):
        s = s[8:]
    return get_variant(s)


def fit_and_attach_stage_c_stats(
    cfg: Any,
    train_data,
    valid_data,
    test_data,
    device: torch.device | str = "cpu",
) -> StageCFeatureStore:
    """Fit on train patients; attach stat_raw [N,3,D] + stat_role_masks [N,3]."""
    variant = _resolve_variant(cfg)
    seed = int(getattr(cfg.data, "candidate_seed", 20260722))
    scenario = str(getattr(cfg.data.dataset, "scenario", "clean"))

    reg = _load_registry(seed)
    context_map = build_context_map(reg)
    triples = build_candidate_coparent_triples(context_map)
    triples_hash = triples_fingerprint(triples)

    patient_df = load_patient_matrix_with_split(cfg)
    train_patients = train_patient_df(patient_df)
    # Leakage guard: never pass val/test patient rows into fit.
    if "split" in patient_df.columns:
        bad = set(train_patients["split"].astype(str).str.lower().unique()) - {
            "train",
            "training",
        }
        if bad:
            raise RuntimeError(
                f"Stage C fit received non-train split labels: {sorted(bad)}"
            )

    splits = (train_data, valid_data, test_data)
    required_pairs = _collect_required_pairs(splits, triples)
    store = fit_stage_c_features(
        train_patients,
        required_pairs,
        variant,
        scenario=scenario,
    )
    d = store.raw_dim

    for data, split_name in zip(splits, ("train", "valid", "test")):
        pairs = candidate_pairs_from_data(data)
        n = len(pairs)
        raw = np.zeros((n, N_ROLES, d), dtype=np.float32)
        masks = np.zeros((n, N_ROLES), dtype=np.float32)

        for i, (u, v) in enumerate(pairs):
            key = (str(u), str(v))
            if key in triples:
                meta = triples[key]
                role_pairs = meta["pairs"]  # AZ, AG, ZG
                structural = (1.0, 1.0, 1.0)
            else:
                # Missing Z → masks [0,1,0]; AG only.
                role_pairs = ((str(u), "__MISSING_Z__"), key, ("__MISSING_Z__", str(v)))
                structural = (0.0, 1.0, 0.0)

            for r, (pu, pv) in enumerate(role_pairs):
                pk = (str(pu), str(pv))
                if structural[r] < 0.5:
                    # Keep zeros; mask 0
                    continue
                vec = store.pair_raw.get(pk)
                applicable = store.pair_applicable.get(pk, False)
                if vec is None:
                    # Should not happen if required_pairs collected correctly
                    continue
                raw[i, r, :] = np.asarray(vec, dtype=np.float32).reshape(-1)[:d]
                # Combine structural presence with feature applicability
                # (HCR_BINARY_ONLY sets applicable=False for non-BB).
                masks[i, r] = 1.0 if applicable else 0.0
                if not applicable:
                    raw[i, r, :] = 0.0

        data.stat_raw = torch.as_tensor(raw, dtype=torch.float32, device=device)
        data.stat_role_masks = torch.as_tensor(masks, dtype=torch.float32, device=device)
        data.stat_raw_dim = int(d)
        data.stat_variant = variant.config_id
        data.stat_pair_roles = list(PAIR_ROLES)
        data.stat_fit_scope = "patient_train_only"
        data.stat_patient_train_fingerprint = store.patient_train_fingerprint
        data.stat_triples_hash = triples_hash
        data.stat_n_train_patients = int(store.n_train_patients)
        data.stat_split_name = split_name
        # Do not precompute g_stat — decoder encodes raw+masks (S0 → zeros).
        if hasattr(data, "g_stat"):
            try:
                delattr(data, "g_stat")
            except Exception:
                data.g_stat = None

    n_with_z = sum(
        1
        for data in splits
        for u, v in candidate_pairs_from_data(data)
        if (str(u), str(v)) in triples
    )
    print(
        f"\nSTAGE C STATS\n  variant: {variant.config_id}\n"
        f"  raw_dim: {d}\n  roles: {list(PAIR_ROLES)}\n"
        f"  missing Z → masks [0,1,0]\n"
        f"  fit_scope: patient_train_only (n={store.n_train_patients})\n"
        f"  fingerprint: {store.patient_train_fingerprint[:16]}…\n"
        f"  triples: {len(triples)}  candidates_with_Z≈{n_with_z}"
    )
    return store
