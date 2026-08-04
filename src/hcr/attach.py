"""Attach leakage-safe HCR features onto Task A HeteroData splits."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from data.candidate_pairs import candidate_pairs_from_data, unique_pairs
from data.patient_matrix import load_patient_matrix_with_split, train_patient_df
from hcr.pair_encoder import HCRPairEncoder
from hcr.variable_spec import VariableSpec
from hcr.variable_specs_v3 import VARIABLE_SPECS


def hcr_enabled(cfg: Any) -> bool:
    hcr_cfg = getattr(cfg, "hcr", None)
    if hcr_cfg is None:
        return False
    return bool(getattr(hcr_cfg, "enabled", False))


def fit_and_attach_hcr(
    cfg: Any,
    train_data,
    valid_data,
    test_data,
    variable_specs: Mapping[str, VariableSpec] | None = None,
    device: torch.device | str = "cpu",
) -> HCRPairEncoder | None:
    """Fit HCR on train patients only; attach features to all candidate splits."""
    if not hcr_enabled(cfg):
        for data in (train_data, valid_data, test_data):
            data.hcr_enabled = False
            data.hcr_features = None
            data.hcr_supported = None
        return None

    specs = dict(variable_specs or VARIABLE_SPECS)
    encoder = HCRPairEncoder.from_hydra(cfg.hcr, specs)

    patient_df = load_patient_matrix_with_split(cfg)
    fit_split = str(getattr(cfg.hcr, "fit_split", "train")).strip().lower()
    if fit_split not in {"train", "training"}:
        raise ValueError(
            f"HCR fit_split must be 'train' (got {fit_split!r}). "
            "Validation/test statistics must never be fitted."
        )
    train_patients = train_patient_df(patient_df)

    all_pairs = unique_pairs(
        candidate_pairs_from_data(train_data)
        + candidate_pairs_from_data(valid_data)
        + candidate_pairs_from_data(test_data)
    )
    encoder.fit(train_patient_df=train_patients, candidate_pairs=all_pairs)

    for data in (train_data, valid_data, test_data):
        pairs = candidate_pairs_from_data(data)
        data.hcr_features = encoder.transform(pairs, device=device)
        data.hcr_supported = torch.as_tensor(
            encoder.supported_mask(pairs), dtype=torch.bool, device=device
        )
        data.hcr_enabled = True
        data.hcr_variant = str(encoder.config.variant)
        data.hcr_dim = int(encoder.config.output_dim)
        data.hcr_fit_split = "train"
        data.hcr_n_train_patients = int(len(train_patients))
        data.hcr_n_binary_pairs = int(encoder.n_binary_pairs)
        data.hcr_n_unsupported_pairs = int(encoder.n_unsupported_pairs)
        data.hcr_append_supported_mask = bool(encoder.config.append_supported_mask)
        data.hcr_force_zero_features = bool(encoder.config.force_zero_features)

    print(
        "\nHCR FEATURES"
        f"\n  variant:            {encoder.config.variant}"
        f"\n  output_dim:         {encoder.config.output_dim}"
        f"\n  append_mask:        {encoder.config.append_supported_mask}"
        f"\n  force_zero:         {encoder.config.force_zero_features}"
        f"\n  fit_split:          train"
        f"\n  n_train_patients:   {len(train_patients)}"
        f"\n  n_unique_pairs:     {len(all_pairs)}"
        f"\n  n_binary_pairs:     {encoder.n_binary_pairs}"
        f"\n  n_unsupported:      {encoder.n_unsupported_pairs}"
    )
    return encoder
