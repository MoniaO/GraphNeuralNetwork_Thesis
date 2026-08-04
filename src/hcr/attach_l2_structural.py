"""Attach structural latent pairwise HCR (24-d) for L2 / Wave 7 V0.

No motif edge hiding — candidates are scored on the full G_train message-passing
graph. Contexts come from Wave 5C registry / co-parents.

Honest motif (frozen since L2 audit):

    AZ ⊕ AG ⊕ ZG   (candidate_coparent_triangle)

Legacy comments called this AB⊕AY⊕BY with Y:=G; that naming was misleading.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from data.candidate_pairs import candidate_pairs_from_data
from data.patient_matrix import load_patient_matrix_with_split, train_patient_df
from hcr.pair_encoder import HCRPairEncoder
from hcr.variable_specs_v3 import VARIABLE_SPECS
from hcr.wave7.motif import (
    CONTEXT_SELECTION_RULE,
    MOTIF_TYPE,
    PAIR_ROLES,
    build_candidate_coparent_triples,
    triples_fingerprint,
)
from wnerw.wave5c.edge_context_registry import build_context_map
from wnerw.wave5c.runtime import ROOT as REPO_ROOT


CONTEXT_DIM = 24


class _HCR2Cfg:
    variant = "binary_compact"
    smoothing = 0.5
    output_dim = 8
    unknown_pair_value = 0.0
    unsupported_pair_mode = "zeros"
    force_zero_features = False
    append_supported_mask = False
    features = None


def _load_context_map(seed: int) -> dict[tuple[str, str], tuple[str, ...]]:
    reg_path = REPO_ROOT / "outputs/wave5c/registry/edge_context_registry.csv"
    if not reg_path.exists():
        return {}
    reg = pd.read_csv(reg_path)
    if "seed" in reg.columns:
        reg = reg[reg["seed"] == int(seed)]
    return build_context_map(reg)


def fit_and_attach_l2_structural(
    cfg: Any,
    train_data,
    valid_data,
    test_data,
    device: torch.device | str = "cpu",
):
    """Fit HCR2 on train patients; attach AZ⊕AG⊕ZG (24) or padded HCR2 (8→24)."""
    seed = int(getattr(cfg.training, "seed", 20260722))
    context_map = _load_context_map(seed)
    triples = build_candidate_coparent_triples(context_map)
    triples_hash = triples_fingerprint(triples)

    patient_df = load_patient_matrix_with_split(cfg)
    train_patients = train_patient_df(patient_df)

    enc = HCRPairEncoder.from_hydra(_HCR2Cfg(), VARIABLE_SPECS)
    splits = (train_data, valid_data, test_data)
    pairs_all: list[tuple[str, str]] = []
    for data in splits:
        pairs_all.extend(candidate_pairs_from_data(data))

    extra: list[tuple[str, str]] = []
    for meta in triples.values():
        extra.extend(list(meta["pairs"]))

    unique = list({(str(u), str(v)) for u, v in pairs_all + extra})
    enc.fit(train_patient_df=train_patients, candidate_pairs=unique)

    cache_rows = []
    for data in splits:
        pairs = candidate_pairs_from_data(data)
        matrix = np.zeros((len(pairs), CONTEXT_DIM), dtype=np.float32)
        supported = []
        motif_available = []
        for i, (u, v) in enumerate(pairs):
            key = (str(u), str(v))
            if key in triples:
                meta_t = triples[key]
                parts = [enc.transform([p]).cpu().numpy()[0] for p in meta_t["pairs"]]
                matrix[i] = np.concatenate(parts).astype(np.float32)
                supported.append(True)
                motif_available.append(True)
                selected_context = meta_t["selected_context"]
                context_candidates = meta_t["context_candidates"]
            else:
                base = enc.transform([key]).cpu().numpy()[0]
                matrix[i, :8] = base[:8]
                supported.append(bool(np.any(np.abs(base[:8]) > 0)))
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
                    "context_candidates": str(context_candidates),
                    "context_selection_rule": CONTEXT_SELECTION_RULE,
                    "pair_roles": str(list(PAIR_ROLES)),
                    **{f"hcr_{j}": float(matrix[i, j]) for j in range(CONTEXT_DIM)},
                }
            )
        data.hcr_features = torch.as_tensor(matrix, dtype=torch.float32, device=device)
        data.hcr_supported = torch.tensor(supported, dtype=torch.bool, device=device)
        data.hcr_motif_available = torch.tensor(motif_available, dtype=torch.bool, device=device)
        data.hcr_enabled = True
        data.hcr_variant = "structural_latent_pairwise_az_ag_zg"
        data.hcr_dim = CONTEXT_DIM
        data.hcr_motif_type = MOTIF_TYPE
        data.hcr_pair_roles = list(PAIR_ROLES)
        data.hcr_triples_hash = triples_hash
        data.hcr_fit_split = "train"
        data.hcr_n_train_patients = int(len(train_patients))

    cache_dir = REPO_ROOT / "outputs/hcr"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "structural_latent_pairwise_az_ag_zg.csv"
    # Keep legacy filename alias for older fingerprinting scripts.
    legacy_path = cache_dir / "structural_latent_pairwise_aby.csv"
    df_cache = pd.DataFrame(cache_rows).drop_duplicates(
        subset=["source", "target"], keep="first"
    )
    df_cache.to_csv(cache_path, index=False)
    df_cache.to_csv(legacy_path, index=False)

    fp_path = REPO_ROOT / "outputs" / "wave7" / "MOTIF_FINGERPRINT_V0.json"
    fp_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    fp_path.write_text(
        json.dumps(
            {
                "motif_type": MOTIF_TYPE,
                "pair_roles": list(PAIR_ROLES),
                "context_selection_rule": CONTEXT_SELECTION_RULE,
                "triples_hash": triples_hash,
                "seed": seed,
                "variant": "W7_V0_L2_BINARY_COMPACT",
            },
            indent=2,
        )
    )

    print(
        "\nHCR L2 STRUCTURAL"
        f"\n  motif: {MOTIF_TYPE} {PAIR_ROLES}"
        f"\n  variant: structural_latent_pairwise_az_ag_zg"
        f"\n  dim: {CONTEXT_DIM}"
        f"\n  contexts: {len(triples)}"
        f"\n  triples_hash: {triples_hash[:12]}…"
        f"\n  cache: {cache_path}"
    )
    return enc
