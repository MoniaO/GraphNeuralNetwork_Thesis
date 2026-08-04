"""Attach HCR for Wave 4C latent-gate recovery (HCR without gate column G).

Hide A→G from G_train (same as Wave 3B). Features for the candidate (A,G) use
downstream outcome Y instead of G:

  L0  none
  L1  concat HCR2(A,B) ‖ HCR2(A,Y) ‖ HCR2(B,Y)
  L2  HCR3(A,B,Y) full
  L3  HCR3(A,B,Y) without a111
  L4  L3 + shuffle Z=B
  L5  L3 + matched-random Z
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from data.candidate_pairs import candidate_pairs_from_data, unique_pairs
from data.patient_matrix import load_patient_matrix_with_split, train_patient_df
from experiments.motif_completion import (
    latent_context_map,
    latent_gate_enabled,
    latent_triple_map,
    motif_completion_enabled,
)
from hcr.attach import fit_and_attach_hcr, hcr_enabled
from hcr.hcr3_encoder import HCR3EncoderConfig, HCR3PairEncoder
from hcr.pair_encoder import HCRPairEncoder
from hcr.registry import HCR3_VARIANTS
from hcr.variable_spec import VariableSpec, VariableType
from hcr.variable_specs_v3 import VARIABLE_SPECS

L1_VARIANT = "latent_pairwise_aby"
L1_DIM = 24  # 3 × binary_compact (8)


class _HCR2Cfg:
    variant = "binary_compact"
    smoothing = 0.5
    output_dim = 8
    unknown_pair_value = 0.0
    unsupported_pair_mode = "zeros"
    force_zero_features = False
    append_supported_mask = False
    features = None


def _is_hcr3(cfg: Any) -> bool:
    if not hcr_enabled(cfg):
        return False
    variant = str(getattr(cfg.hcr, "variant", "")).strip().lower()
    return variant in HCR3_VARIANTS


def _is_l1(cfg: Any) -> bool:
    if not hcr_enabled(cfg):
        return False
    return str(getattr(cfg.hcr, "variant", "")).strip().lower() == L1_VARIANT


def fit_and_attach_hcr_wave4c(
    cfg: Any,
    train_data,
    valid_data,
    test_data,
    variable_specs: Mapping[str, VariableSpec] | None = None,
    device: torch.device | str = "cpu",
):
    """Wave 4C attach — never reads gate G patient values for motif features."""
    if not latent_gate_enabled(cfg):
        raise RuntimeError("fit_and_attach_hcr_wave4c requires experiment.latent_gate.enabled")
    if not motif_completion_enabled(cfg):
        raise RuntimeError("Wave 4C requires motif_completion.enabled (hide A→G)")

    if not hcr_enabled(cfg):
        return fit_and_attach_hcr(
            cfg, train_data, valid_data, test_data, variable_specs, device
        )

    if _is_l1(cfg):
        return _attach_l1(cfg, train_data, valid_data, test_data, variable_specs, device)

    if not _is_hcr3(cfg):
        # Fallback pairwise (should not be used for L2–L5).
        return fit_and_attach_hcr(
            cfg, train_data, valid_data, test_data, variable_specs, device
        )

    return _attach_hcr3_latent(
        cfg, train_data, valid_data, test_data, variable_specs, device
    )


def _attach_l1(cfg, train_data, valid_data, test_data, variable_specs, device):
    specs = dict(variable_specs or VARIABLE_SPECS)
    patient_df = load_patient_matrix_with_split(cfg)
    train_patients = train_patient_df(patient_df)
    ctx = latent_context_map(cfg, train_patients)

    h2_cfg = _HCR2Cfg()
    h2_cfg.smoothing = float(getattr(cfg.hcr, "smoothing", 0.5))
    hcr2 = HCRPairEncoder.from_hydra(h2_cfg, specs)
    # Fit on all pairs that appear in concat + all candidates.
    extra_pairs = []
    for meta in ctx.values():
        extra_pairs.extend(
            [
                (meta["A"], meta["B"]),
                (meta["A"], meta["Y"]),
                (meta["B"], meta["Y"]),
            ]
        )
    all_pairs = unique_pairs(
        candidate_pairs_from_data(train_data)
        + candidate_pairs_from_data(valid_data)
        + candidate_pairs_from_data(test_data)
        + extra_pairs
    )
    hcr2.fit(train_patient_df=train_patients, candidate_pairs=all_pairs)

    def one_vec(pair):
        if pair not in ctx:
            base = hcr2.transform([pair]).cpu().numpy()[0]
            vec = np.zeros((L1_DIM,), dtype=np.float32)
            vec[: min(len(base), 8)] = base[:8]
            return vec
        meta = ctx[pair]
        parts = [
            hcr2.transform([(meta["A"], meta["B"])]).cpu().numpy()[0],
            hcr2.transform([(meta["A"], meta["Y"])]).cpu().numpy()[0],
            hcr2.transform([(meta["B"], meta["Y"])]).cpu().numpy()[0],
        ]
        return np.concatenate(parts, axis=0).astype(np.float32)

    for data in (train_data, valid_data, test_data):
        pairs = candidate_pairs_from_data(data)
        matrix = np.stack([one_vec(p) for p in pairs], axis=0)
        data.hcr_features = torch.as_tensor(matrix, dtype=torch.float32, device=device)
        data.hcr_supported = torch.tensor(
            [p in ctx for p in pairs], dtype=torch.bool, device=device
        )
        data.hcr_enabled = True
        data.hcr_variant = L1_VARIANT
        data.hcr_dim = L1_DIM
        data.hcr_fit_split = "train"
        data.hcr_n_train_patients = int(len(train_patients))
        data.latent_gate_pairs = list(ctx.keys())

    # Fake encoder-like object for decoder dim alignment.
    class _L1Enc:
        class config:
            output_dim = L1_DIM
            variant = L1_VARIANT

    print(
        "\nHCR L1 FEATURES (Wave 4C latent pairwise A-B|A-Y|B-Y)"
        f"\n  output_dim: {L1_DIM}"
        f"\n  motif pairs: {len(ctx)}"
        f"\n  gate columns used: none"
    )
    return _L1Enc()


def _attach_hcr3_latent(cfg, train_data, valid_data, test_data, variable_specs, device):
    specs = dict(variable_specs or VARIABLE_SPECS)
    patient_df = load_patient_matrix_with_split(cfg)
    train_patients = train_patient_df(patient_df)

    include_a111 = bool(getattr(cfg.hcr, "include_a111", True))
    shuffle_z = bool(getattr(cfg.hcr, "shuffle_z", False))
    variant = str(cfg.hcr.variant).strip().lower()
    hcr3_dim = int(getattr(cfg.hcr, "output_dim", 17))

    triples = latent_triple_map(cfg, train_patients)

    encoder = HCR3PairEncoder(
        HCR3EncoderConfig(
            output_dim=hcr3_dim,
            smoothing=float(getattr(cfg.hcr, "smoothing", 0.5)),
            include_a111=include_a111,
            shuffle_z=shuffle_z,
            shuffle_seed=int(getattr(cfg.hcr, "shuffle_seed", 20260722)),
            variant=variant,
        )
    )
    # Leakage check is per-candidate: gate G = pair[1] must not appear in (A,Y,B).
    encoder.fit_triples(
        train_patient_df=train_patients,
        candidate_triples=triples,
        forbidden_columns=None,
    )

    h2_cfg = _HCR2Cfg()
    h2_cfg.smoothing = float(getattr(cfg.hcr, "smoothing", 0.5))
    hcr2 = HCRPairEncoder.from_hydra(h2_cfg, specs)
    all_pairs = unique_pairs(
        candidate_pairs_from_data(train_data)
        + candidate_pairs_from_data(valid_data)
        + candidate_pairs_from_data(test_data)
    )
    hcr2.fit(train_patient_df=train_patients, candidate_pairs=all_pairs)

    motif_keys = set(triples.keys())

    def build_matrix(pairs):
        rows = []
        for pair in pairs:
            if pair in motif_keys:
                vec = encoder.transform([pair]).cpu().numpy()[0]
            else:
                base = hcr2.transform([pair]).cpu().numpy()[0]
                vec = np.zeros((hcr3_dim,), dtype=np.float32)
                n = min(len(base), hcr3_dim)
                vec[:n] = base[:n]
            rows.append(vec)
        return torch.as_tensor(np.stack(rows, axis=0), dtype=torch.float32, device=device)

    for data in (train_data, valid_data, test_data):
        pairs = candidate_pairs_from_data(data)
        data.hcr_features = build_matrix(pairs)
        data.hcr_supported = torch.tensor(
            [p in motif_keys for p in pairs],
            dtype=torch.bool,
            device=device,
        )
        data.hcr_enabled = True
        data.hcr_variant = variant
        data.hcr_dim = hcr3_dim
        data.hcr_fit_split = "train"
        data.hcr_n_train_patients = int(len(train_patients))
        data.hcr_n_motif_z = int(encoder.n_with_z)
        data.latent_gate_triples = {f"{a}->{g}": triples[(a, g)] for a, g in motif_keys}

    sample = next(iter(triples.values())) if triples else None
    gates = sorted({g for (_a, g) in triples})
    print(
        "\nHCR-3 FEATURES (Wave 4C latent-gate)"
        f"\n  variant:      {variant}"
        f"\n  output_dim:   {hcr3_dim}"
        f"\n  motif triples:{encoder.n_with_z}"
        f"\n  shuffle_z:    {shuffle_z}"
        f"\n  include_a111: {include_a111}"
        f"\n  sample (A,Y,B): {sample}"
        f"\n  candidate gates (not in features): {gates}"
    )
    return encoder
