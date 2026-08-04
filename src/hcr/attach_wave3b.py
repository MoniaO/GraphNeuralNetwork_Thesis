"""Attach HCR-2 or HCR-3 features for Wave 3B motif-completion runs."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from data.candidate_pairs import candidate_pairs_from_data, unique_pairs
from data.patient_matrix import load_patient_matrix_with_split, train_patient_df
from experiments.motif_completion import motif_completion_enabled, oracle_z_map
from hcr.attach import fit_and_attach_hcr, hcr_enabled
from hcr.hcr3_encoder import HCR3EncoderConfig, HCR3PairEncoder
from hcr.pair_encoder import HCRPairEncoder
from hcr.registry import HCR3_VARIANTS
from hcr.variable_spec import VariableSpec, VariableType
from hcr.variable_specs_v3 import VARIABLE_SPECS


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


def fit_and_attach_hcr_wave3b(
    cfg: Any,
    train_data,
    valid_data,
    test_data,
    variable_specs: Mapping[str, VariableSpec] | None = None,
    device: torch.device | str = "cpu",
):
    """Dispatch HCR-2 vs HCR-3; HCR-3 uses oracle co-parent Z on held-out edges."""
    if not hcr_enabled(cfg) or not _is_hcr3(cfg):
        return fit_and_attach_hcr(
            cfg, train_data, valid_data, test_data, variable_specs, device
        )

    specs = dict(variable_specs or VARIABLE_SPECS)
    patient_df = load_patient_matrix_with_split(cfg)
    train_patients = train_patient_df(patient_df)

    include_a111 = bool(getattr(cfg.hcr, "include_a111", True))
    shuffle_z = bool(getattr(cfg.hcr, "shuffle_z", False))
    variant = str(cfg.hcr.variant).strip().lower()
    hcr3_dim = int(getattr(cfg.hcr, "output_dim", 17))

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

    z_map = oracle_z_map(cfg, train_patients) if motif_completion_enabled(cfg) else {}
    encoder.fit(train_patient_df=train_patients, candidate_z=z_map)

    h2_cfg = _HCR2Cfg()
    h2_cfg.smoothing = float(getattr(cfg.hcr, "smoothing", 0.5))
    hcr2 = HCRPairEncoder.from_hydra(h2_cfg, specs)
    all_pairs = unique_pairs(
        candidate_pairs_from_data(train_data)
        + candidate_pairs_from_data(valid_data)
        + candidate_pairs_from_data(test_data)
    )
    hcr2.fit(train_patient_df=train_patients, candidate_pairs=all_pairs)

    motif_keys = set(z_map.keys())

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
            [
                (
                    s in specs
                    and t in specs
                    and specs[s].variable_type == VariableType.BINARY
                    and specs[t].variable_type == VariableType.BINARY
                )
                for s, t in pairs
            ],
            dtype=torch.bool,
            device=device,
        )
        data.hcr_enabled = True
        data.hcr_variant = variant
        data.hcr_dim = hcr3_dim
        data.hcr_fit_split = "train"
        data.hcr_n_train_patients = int(len(train_patients))
        data.hcr_n_motif_z = int(encoder.n_with_z)
        data.motif_z_pairs = list(motif_keys)

    print(
        "\nHCR-3 FEATURES (Wave 3B)"
        f"\n  variant:      {variant}"
        f"\n  output_dim:   {hcr3_dim}"
        f"\n  motif Z fit:  {encoder.n_with_z}"
        f"\n  shuffle_z:    {shuffle_z}"
        f"\n  include_a111: {include_a111}"
    )
    return encoder
