"""Attach HCR for Wave 4D — structural context selection + role audit.

Variants:
  D0  none
  D1  structural_latent_pairwise     — Z from Pa_Gtrain(G)\\{X}
  D2  all_context_top1               — unrestricted scan (unsafe control)
  D3  structural_context_shuffled    — structural Z, shuffled patient values
  D4  matched_random_context         — prevalence/layer-matched random Z
  D5  hcr3_selected_capacity_matched — HCR3 w/o a111, zero-padded 17→24
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from data.candidate_pairs import candidate_pairs_from_data, unique_pairs
from data.patient_matrix import load_patient_matrix_with_split, train_patient_df
from experiments.motif_completion import (
    hide_tasks,
    motif_completion_enabled,
)
from hcr.attach import fit_and_attach_hcr, hcr_enabled
from hcr.hcr3_encoder import HCR3EncoderConfig, HCR3PairEncoder
from hcr.motif_registry_v3 import GATE_OUTCOMES
from hcr.pair_encoder import HCRPairEncoder
from hcr.structural_context import (
    StructuralSelectorAudit,
    load_node_metadata,
    matched_random_context,
    select_all_context_top1_map,
    select_structural_context_map,
)
from hcr.variable_spec import VariableSpec
from hcr.variable_specs_v3 import VARIABLE_SPECS

CONTEXT_DIM = 24
D1_VARIANT = "structural_latent_pairwise"
D2_VARIANT = "all_context_top1"
D3_VARIANT = "structural_context_shuffled"
D4_VARIANT = "matched_random_context"
D5_VARIANT = "hcr3_selected_capacity_matched"

PAIRWISE_VARIANTS = {D1_VARIANT, D2_VARIANT, D3_VARIANT, D4_VARIANT, "latent_pairwise_aby"}


class _HCR2Cfg:
    variant = "binary_compact"
    smoothing = 0.5
    output_dim = 8
    unknown_pair_value = 0.0
    unsupported_pair_mode = "zeros"
    force_zero_features = False
    append_supported_mask = False
    features = None


def pad_context_features(
    features: torch.Tensor,
    target_dim: int = CONTEXT_DIM,
) -> torch.Tensor:
    if features.size(-1) > target_dim:
        raise ValueError("Feature dimension exceeds target.")
    pad_dim = target_dim - features.size(-1)
    if pad_dim == 0:
        return features
    padding = torch.zeros(
        features.size(0),
        pad_dim,
        dtype=features.dtype,
        device=features.device,
    )
    return torch.cat([features, padding], dim=-1)


def wave4d_enabled(cfg: Any) -> bool:
    exp = getattr(cfg, "experiment", None)
    if exp is None:
        return False
    wave = str(getattr(exp, "wave", "")).upper()
    if "WAVE4D" in wave or "CONTEXT_ROLE" in wave:
        return True
    cra = getattr(exp, "causal_role_audit", None)
    return bool(cra is not None and getattr(cra, "enabled", False))


def _variant(cfg: Any) -> str:
    return str(getattr(cfg.hcr, "variant", "none")).strip().lower()


def _context_mode(cfg: Any) -> str:
    cs = getattr(cfg, "context_selection", None)
    if cs is None:
        exp = getattr(cfg, "experiment", None)
        cs = getattr(exp, "context_selection", None) if exp is not None else None
    if cs is None:
        # Infer from variant name.
        v = _variant(cfg)
        if v == D2_VARIANT:
            return "all_context_top1"
        if v == D4_VARIANT:
            return "matched_random"
        if v in {D1_VARIANT, D3_VARIANT, D5_VARIANT, "latent_pairwise_aby"}:
            return "structural"
        return "none"
    return str(getattr(cs, "mode", "structural")).strip().lower()


def _assert_no_leakage(cfg: Any, selected: dict, train_patients) -> None:
    audit = StructuralSelectorAudit()
    assert audit.graph_source == "train"
    assert not audit.has_access_to_true_graph
    assert not audit.uses_test_labels

    cs = getattr(cfg, "context_selection", None)
    if cs is None:
        exp = getattr(cfg, "experiment", None)
        cs = getattr(exp, "context_selection", None) if exp is not None else None
    if cs is not None:
        assert str(getattr(cs, "source_graph", "train")) == "train"
        assert not bool(getattr(cs, "use_true_graph", False))

    for (src, gate), z in selected.items():
        y = GATE_OUTCOMES.get(gate)
        hcr_feature_columns = {c for c in (src, z, y) if c}
        assert gate not in hcr_feature_columns, f"gate {gate} leaked into HCR columns"

    assert "role_label" not in str(getattr(cfg, "model", {}))


def resolve_context_map(cfg: Any, train_data, train_patients) -> dict[tuple[str, str], str | None]:
    """Select Z per held-out (X,G) according to context_selection.mode / variant."""
    tasks = hide_tasks(cfg, train_patients)
    mode = _context_mode(cfg)
    variant = _variant(cfg)
    seed = int(getattr(cfg.hcr, "shuffle_seed", getattr(cfg.training, "seed", 20260722)))
    meta = load_node_metadata(cfg)
    structural = select_structural_context_map(cfg, train_data, tasks)

    if variant == D2_VARIANT or mode in {"all_context_top1", "unrestricted"}:
        return select_all_context_top1_map(cfg, train_patients, tasks)

    if variant == D4_VARIANT or mode in {"matched_random", "matched_random_context"}:
        out: dict[tuple[str, str], str | None] = {}
        for t in tasks:
            key = (t.candidate_source, t.candidate_target)
            y = GATE_OUTCOMES.get(t.gate) or (t.children[0] if t.children else None)
            out[key] = matched_random_context(
                t.candidate_source,
                t.gate,
                y,
                structural.get(key),
                meta,
                seed,
            )
        return out

    # D1 / D3 / D5 — structural co-parent from G_train.
    return structural


def _triple_map_from_contexts(
    cfg: Any,
    contexts: dict[tuple[str, str], str | None],
    samples=None,
) -> dict[tuple[str, str], tuple[str, str, str]]:
    tasks = hide_tasks(cfg, samples)
    out: dict[tuple[str, str], tuple[str, str, str]] = {}
    for t in tasks:
        key = (t.candidate_source, t.candidate_target)
        y = GATE_OUTCOMES.get(t.gate) or (t.children[0] if t.children else None)
        z = contexts.get(key)
        if y is None or z is None:
            continue
        # Leakage hard-stop: gate must not be an HCR column.
        assert t.gate not in {t.candidate_source, y, z}
        out[key] = (t.candidate_source, y, z)
    return out


def fit_and_attach_hcr_wave4d(
    cfg: Any,
    train_data,
    valid_data,
    test_data,
    variable_specs: Mapping[str, VariableSpec] | None = None,
    device: torch.device | str = "cpu",
):
    if not wave4d_enabled(cfg):
        raise RuntimeError("fit_and_attach_hcr_wave4d requires Wave 4D experiment flag")
    if not motif_completion_enabled(cfg):
        raise RuntimeError("Wave 4D requires motif_completion.enabled")

    if not hcr_enabled(cfg):
        return fit_and_attach_hcr(
            cfg, train_data, valid_data, test_data, variable_specs, device
        )

    variant = _variant(cfg)
    patient_df = load_patient_matrix_with_split(cfg)
    train_patients = train_patient_df(patient_df)
    contexts = resolve_context_map(cfg, train_data, train_patients)
    _assert_no_leakage(cfg, contexts, train_patients)

    if variant == D5_VARIANT or str(getattr(cfg.hcr, "capacity_matched", False)).lower() == "true":
        return _attach_d5(
            cfg, train_data, valid_data, test_data, train_patients, contexts, variable_specs, device
        )

    if variant in PAIRWISE_VARIANTS or _context_mode(cfg) != "none":
        return _attach_pairwise(
            cfg, train_data, valid_data, test_data, train_patients, contexts, variable_specs, device
        )

    return fit_and_attach_hcr(
        cfg, train_data, valid_data, test_data, variable_specs, device
    )


def _attach_pairwise(
    cfg, train_data, valid_data, test_data, train_patients, contexts, variable_specs, device
):
    specs = dict(variable_specs or VARIABLE_SPECS)
    variant = _variant(cfg)
    shuffle = variant == D3_VARIANT or bool(getattr(cfg.hcr, "shuffle_z", False))
    seed = int(getattr(cfg.hcr, "shuffle_seed", getattr(cfg.training, "seed", 20260722)))

    h2_cfg = _HCR2Cfg()
    h2_cfg.smoothing = float(getattr(cfg.hcr, "smoothing", 0.5))
    hcr2 = HCRPairEncoder.from_hydra(h2_cfg, specs)

    triples = _triple_map_from_contexts(cfg, contexts, train_patients)
    extra_pairs = []
    for a, y, z in triples.values():
        extra_pairs.extend([(a, z), (a, y), (z, y)])
    all_pairs = unique_pairs(
        candidate_pairs_from_data(train_data)
        + candidate_pairs_from_data(valid_data)
        + candidate_pairs_from_data(test_data)
        + extra_pairs
    )

    # Fit on (possibly shuffled) train matrix.
    fit_df = train_patients
    if shuffle:
        fit_df = train_patients.copy()
        rng = np.random.default_rng(seed)
        # Shuffle each context column independently among train patients.
        for z in sorted({t[2] for t in triples.values()}):
            if z in fit_df.columns:
                fit_df[z] = rng.permutation(fit_df[z].to_numpy())

    hcr2.fit(train_patient_df=fit_df, candidate_pairs=all_pairs)

    def one_vec(pair):
        if pair not in triples:
            base = hcr2.transform([pair]).cpu().numpy()[0]
            vec = np.zeros((CONTEXT_DIM,), dtype=np.float32)
            vec[: min(len(base), 8)] = base[:8]
            return vec
        a, y, z = triples[pair]
        parts = [
            hcr2.transform([(a, z)]).cpu().numpy()[0],
            hcr2.transform([(a, y)]).cpu().numpy()[0],
            hcr2.transform([(z, y)]).cpu().numpy()[0],
        ]
        return np.concatenate(parts, axis=0).astype(np.float32)

    ctx_serial = {f"{a}->{g}": z for (a, g), z in contexts.items()}
    for data in (train_data, valid_data, test_data):
        pairs = candidate_pairs_from_data(data)
        matrix = np.stack([one_vec(p) for p in pairs], axis=0)
        data.hcr_features = torch.as_tensor(matrix, dtype=torch.float32, device=device)
        data.hcr_supported = torch.tensor(
            [p in triples for p in pairs], dtype=torch.bool, device=device
        )
        data.hcr_enabled = True
        data.hcr_variant = variant
        data.hcr_dim = CONTEXT_DIM
        data.hcr_fit_split = "train"
        data.hcr_n_train_patients = int(len(train_patients))
        data.wave4d_contexts = ctx_serial
        data.wave4d_triples = {f"{a}->{g}": triples[(a, g)] for a, g in triples}

    class _Enc:
        pass

    _Enc.config = type(
        "config",
        (),
        {"output_dim": CONTEXT_DIM, "variant": variant},
    )()

    n_ok = sum(1 for z in contexts.values() if z is not None)
    print(
        "\nHCR WAVE4D PAIRWISE"
        f"\n  variant:      {variant}"
        f"\n  context_mode: {_context_mode(cfg)}"
        f"\n  output_dim:   {CONTEXT_DIM}"
        f"\n  contexts ok:  {n_ok}/{len(contexts)}"
        f"\n  shuffle_z:    {shuffle}"
        f"\n  gate columns used: none"
    )
    return _Enc()


def _attach_d5(
    cfg, train_data, valid_data, test_data, train_patients, contexts, variable_specs, device
):
    specs = dict(variable_specs or VARIABLE_SPECS)
    triples = _triple_map_from_contexts(cfg, contexts, train_patients)
    encoder = HCR3PairEncoder(
        HCR3EncoderConfig(
            output_dim=17,
            smoothing=float(getattr(cfg.hcr, "smoothing", 0.5)),
            include_a111=False,
            shuffle_z=False,
            shuffle_seed=int(getattr(cfg.hcr, "shuffle_seed", 20260722)),
            variant=D5_VARIANT,
        )
    )
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
    ctx_serial = {f"{a}->{g}": z for (a, g), z in contexts.items()}

    def build_matrix(pairs):
        rows = []
        for pair in pairs:
            if pair in motif_keys:
                vec = encoder.transform([pair])
                vec = pad_context_features(vec, CONTEXT_DIM).cpu().numpy()[0]
            else:
                base = hcr2.transform([pair]).cpu().numpy()[0]
                vec = np.zeros((CONTEXT_DIM,), dtype=np.float32)
                n = min(len(base), 8)
                vec[:n] = base[:n]
            rows.append(vec)
        return torch.as_tensor(np.stack(rows, axis=0), dtype=torch.float32, device=device)

    for data in (train_data, valid_data, test_data):
        pairs = candidate_pairs_from_data(data)
        data.hcr_features = build_matrix(pairs)
        data.hcr_supported = torch.tensor(
            [p in motif_keys for p in pairs], dtype=torch.bool, device=device
        )
        data.hcr_enabled = True
        data.hcr_variant = D5_VARIANT
        data.hcr_dim = CONTEXT_DIM
        data.hcr_fit_split = "train"
        data.hcr_n_train_patients = int(len(train_patients))
        data.wave4d_contexts = ctx_serial
        data.wave4d_triples = {f"{a}->{g}": triples[(a, g)] for a, g in motif_keys}

    class _Enc:
        pass

    _Enc.config = type(
        "config",
        (),
        {"output_dim": CONTEXT_DIM, "variant": D5_VARIANT},
    )()

    print(
        "\nHCR WAVE4D D5 CAPACITY-MATCHED"
        f"\n  raw_hcr3_dim: 17 → padded {CONTEXT_DIM}"
        f"\n  motif triples: {encoder.n_with_z}"
        f"\n  gate columns used: none"
    )
    return _Enc()
