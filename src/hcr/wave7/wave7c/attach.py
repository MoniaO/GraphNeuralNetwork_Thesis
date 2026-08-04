"""Attach B2 motifs for Wave 7C architecture/context audits."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from data.candidate_pairs import candidate_pairs_from_data
from data.patient_matrix import load_patient_matrix_with_split, train_patient_df
from hcr.variable_spec import VariableType
from hcr.variable_specs_v3 import VARIABLE_SPECS
from hcr.wave7.motif import (
    PAIR_ROLES,
    build_candidate_coparent_triples,
    triples_fingerprint,
)
from hcr.wave7.panel_b.encoder import PanelBConfig, PanelBPairEncoder
from hcr.wave7.panel_b.packing import PAIR_DIM
from hcr.wave7.panel_b_residual.conditional_edge_gain import (
    TRIPLE_DIM,
    ConditionalEdgeGainResult,
    compute_conditional_edge_gain,
)
from hcr.wave7.panel_b_residual.constants import MOTIF_B2_DIM, N_ROLES
from wnerw.wave5c.edge_context_registry import build_context_map
from wnerw.wave5c.runtime import ROOT as REPO_ROOT

B2_VARIANT = "W7B_B2_JITTER_GHCR_ENRICHED40"


def wave7c_enabled(cfg: Any) -> bool:
    wave = str(getattr(cfg.experiment, "wave", "")).upper()
    variant = str(
        getattr(cfg.experiment, "variant", None) or getattr(cfg.hcr, "variant", "")
    ).upper()
    return (
        variant.startswith("W7C_")
        or variant.startswith("W7D_")
        or variant.startswith("W9_")
        or variant.startswith("W10_")
        or variant.startswith("TASKA_")
        or "WAVE7C" in wave
        or "WAVE7D" in wave
        or "WAVE9" in wave
        or "WAVE10" in wave
        or "TASKA_MLP_VS_KAN" in wave
        or str(getattr(cfg.hcr, "encoder", "")).lower() == "wave7c_b2_audit"
    )


def _is_binary_triple(a: str, z: str, g: str) -> bool:
    return _kind(a) == "binary" and _kind(z) == "binary" and _kind(g) == "binary"


def _column_values(train_df: pd.DataFrame, name: str) -> np.ndarray | None:
    spec = VARIABLE_SPECS.get(str(name))
    col = spec.column_name if spec is not None else str(name)
    if col not in train_df.columns:
        return None
    return pd.to_numeric(train_df[col], errors="coerce").to_numpy(dtype=float)


def _wants_triple(cfg: Any, variant: str) -> bool:
    v = variant.upper()
    if "T1_" in v or "CONDITIONAL_EDGE" in v or "PLUS_CONDITIONAL" in v:
        return True
    dec = getattr(getattr(cfg, "model", None), "decoder", None)
    if dec is not None and getattr(dec, "use_triple", None) is not None:
        return bool(dec.use_triple)
    return False


def _zero_triple(reason: str) -> ConditionalEdgeGainResult:
    return ConditionalEdgeGainResult(
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
        reason=reason,
    )


def _variant(cfg: Any) -> str:
    return str(
        getattr(cfg.experiment, "variant", None)
        or getattr(cfg.hcr, "variant", "W7C_A0_R1_SHARED_PAIR_ENCODER")
    ).strip()


def _kind(name: str) -> str:
    spec = VARIABLE_SPECS.get(str(name))
    if spec is None:
        return "unknown"
    if spec.variable_type == VariableType.BINARY:
        return "binary"
    if spec.variable_type == VariableType.COUNT:
        return "count"
    return "continuous"


def _load_registry(seed: int) -> pd.DataFrame:
    path = REPO_ROOT / "outputs/wave5c/registry/edge_context_registry.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Wave 5C edge-context registry: {path}. "
            "Build it with: PYTHONPATH=src python scripts/build_wave5c_context_registry.py "
            "(requires Wave 5 evidence under outputs/wave5/evidence/). "
            "Without this file Wave 7C/11 silently falls back to AG-only motifs."
        )
    reg = pd.read_csv(path)
    if "seed" in reg.columns:
        reg = reg[reg["seed"] == int(seed)]
    if len(reg) == 0:
        raise ValueError(
            f"Wave 5C registry at {path} has no rows for seed={seed}. "
            "Rebuild with: PYTHONPATH=src python scripts/build_wave5c_context_registry.py"
        )
    return reg


def _registry_meta(reg: pd.DataFrame) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    if reg is None or len(reg) == 0:
        return out
    for row in reg.itertuples(index=False):
        out[(str(row.source), str(row.target))] = {
            "selector_source": str(getattr(row, "selector_source", "")),
            "context_nodes": getattr(row, "context_nodes", "[]"),
        }
    return out


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


def _matched_z_prime(
    a: str,
    g: str,
    z: str,
    *,
    all_nodes: list[str],
    node_degree: dict[str, int],
    available_cols: set[str],
    pair_metas: dict[tuple[str, str], dict],
    rng: np.random.Generator,
) -> str:
    """Pick incorrect Z' matched on type/degree/availability/support.

    Never uses labels or G_true.
    """
    z_type = _kind(z)
    z_deg = node_degree.get(z, 0)
    z_avail = z in available_cols
    az_sup = float(pair_metas.get((a, z), {}).get("n_complete", 0) or 0)
    zg_sup = float(pair_metas.get((z, g), {}).get("n_complete", 0) or 0)

    candidates = [
        n
        for n in all_nodes
        if n not in {a, g, z} and _kind(n) == z_type
    ]
    if not candidates:
        candidates = [n for n in all_nodes if n not in {a, g, z}]
    if not candidates:
        return z

    def _score(n: str) -> tuple:
        avail_pen = 0 if ((n in available_cols) == z_avail) else 1
        deg_pen = abs(node_degree.get(n, 0) - z_deg)
        az_n = float(pair_metas.get((a, n), {}).get("n_complete", 0) or 0)
        zg_n = float(pair_metas.get((n, g), {}).get("n_complete", 0) or 0)
        # Prefer similar support; unknown pairs get large penalty.
        sup_pen = abs(az_n - az_sup) + abs(zg_n - zg_sup)
        if (a, n) not in pair_metas and (n, g) not in pair_metas:
            sup_pen += 1e6
        return (avail_pen, deg_pen, sup_pen, n)

    scored = sorted(candidates, key=_score)
    band = scored[: max(1, min(5, len(scored)))]
    return str(band[int(rng.integers(0, len(band)))])


def fit_and_attach_wave7c(
    cfg: Any,
    train_data,
    valid_data,
    test_data,
    device: torch.device | str = "cpu",
):
    variant = _variant(cfg)
    v_up = variant.upper()
    seed = int(getattr(cfg.data, "candidate_seed", getattr(cfg.training, "seed", 20260722)))
    use_matched_z = "C4_" in v_up or "MATCHED_SHUFFLED" in v_up
    use_permute = "C5_" in v_up or "PATIENT_PERMUTED" in v_up
    use_triple = _wants_triple(cfg, variant)

    reg = _load_registry(seed)
    reg_meta = _registry_meta(reg)
    context_map = build_context_map(reg)
    triples = build_candidate_coparent_triples(context_map)
    triples_hash = triples_fingerprint(triples)

    patient_df = load_patient_matrix_with_split(cfg)
    train_patients = train_patient_df(patient_df)
    scenario = str(getattr(cfg.data.dataset, "scenario", "clean"))

    b2_cfg = PanelBConfig.from_hydra(cfg.hcr, getattr(cfg, "experiment", None))
    b2_cfg.variant = B2_VARIANT
    encoder = PanelBPairEncoder(b2_cfg)
    encoder.scenario = scenario

    splits = (train_data, valid_data, test_data)
    required_pairs = _collect_required_pairs(splits, triples)
    unique_nodes = {n for uv in required_pairs for n in uv}
    encoder.fit(train_patients, unique_nodes)

    # Optional C5: permute each patient-matrix column (fixed seed).
    # Preserves per-variable marginals/missingness; destroys all joints.
    train_work = train_patients.copy()
    if use_permute:
        rng = np.random.default_rng(seed + 17)
        for col in train_work.columns:
            if col in {"patient_id", "split"}:
                continue
            vals = train_work[col].to_numpy().copy()
            idx = np.arange(len(vals))
            rng.shuffle(idx)
            train_work[col] = vals[idx]
        # Re-bind encoder train frame for transform (bases already fit on true train).
        encoder._train_df = train_work.reset_index(drop=True)
        if "patient_id" in encoder._train_df.columns:
            encoder._patient_ids = encoder._train_df["patient_id"].astype(str).to_numpy()
        encoder._pair_cache = {}

    pair_vectors: dict[tuple[str, str], np.ndarray] = {}
    pair_metas: dict[tuple[str, str], dict] = {}
    for u, v in required_pairs:
        vec, meta = encoder.transform_pair(u, v)
        pair_vectors[(str(u), str(v))] = vec
        pair_metas[(str(u), str(v))] = meta

    # Degrees for C4 matching: count appearances as context or endpoint in registry.
    node_degree: dict[str, int] = {}
    for (a, g), meta in triples.items():
        for n in (a, g, meta["Z"]):
            node_degree[str(n)] = node_degree.get(str(n), 0) + 1
    all_nodes = sorted(unique_nodes | set(node_degree))
    available_cols = {
        str(c) for c in train_patients.columns if c not in {"patient_id", "split"}
    }
    rng_c4 = np.random.default_rng(seed + 41)

    # Possibly remap triples for C4.
    triples_use = triples
    if use_matched_z:
        remapped = {}
        for (a, g), meta in triples.items():
            z = str(meta["Z"])
            zp = _matched_z_prime(
                str(a),
                str(g),
                z,
                all_nodes=all_nodes,
                node_degree=node_degree,
                available_cols=available_cols,
                pair_metas=pair_metas,
                rng=rng_c4,
            )
            # Ensure Z' pair vectors exist.
            for pk in ((str(a), zp), (zp, str(g))):
                if pk not in pair_vectors:
                    vec, meta_p = encoder.transform_pair(pk[0], pk[1])
                    pair_vectors[pk] = vec
                    pair_metas[pk] = meta_p
            remapped[(str(a), str(g))] = {
                **meta,
                "Z": zp,
                "pairs": ((str(a), zp), (str(a), str(g)), (zp, str(g))),
                "selected_context": zp,
                "original_Z": z,
                "context_selection_reason": "matched_shuffled_z",
            }
        triples_use = remapped

    # Optional T1: binary–binary–binary I(A;G|Z). Never falls back to I(A;G).
    triple_cache: dict[tuple[str, str, str], ConditionalEdgeGainResult] = {}
    n_train = int(len(train_patients))
    if use_triple:
        for (a_name, g_name), meta_t in triples_use.items():
            z_name = str(meta_t["Z"])
            tkey = (str(a_name), str(z_name), str(g_name))
            if tkey in triple_cache:
                continue
            if not _is_binary_triple(a_name, z_name, g_name):
                triple_cache[tkey] = _zero_triple("non_binary_triple")
                continue
            av = _column_values(train_patients, a_name)
            zv = _column_values(train_patients, z_name)
            gv = _column_values(train_patients, g_name)
            if av is None or zv is None or gv is None:
                triple_cache[tkey] = _zero_triple("missing_column")
            else:
                triple_cache[tkey] = compute_conditional_edge_gain(
                    av, zv, gv, n_train=n_train
                )
        audit_dir = REPO_ROOT / "outputs" / "wave7" / "wave7c" / "etap3" / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_rows = []
        for (a, z, g), res in sorted(triple_cache.items()):
            audit_rows.append(
                {
                    "A": a,
                    "Z": z,
                    "G": g,
                    "n_complete": res.n_complete,
                    "conditional_edge_gain": res.ig,
                    "normalized_edge_gain": res.ig_norm,
                    "conditional_RD": res.rd_cond,
                    "interaction_heterogeneity": res.interaction,
                    "support": res.support,
                    "triple_supported_mask": res.mask,
                    "reason": res.reason,
                }
            )
        pd.DataFrame(audit_rows).to_csv(
            audit_dir / f"CONDITIONAL_EDGE_GAIN_{variant}__seed{seed}.csv",
            index=False,
        )

    for data, split_name in zip(splits, ("train", "valid", "test")):
        pairs = candidate_pairs_from_data(data)
        matrix = np.zeros((len(pairs), MOTIF_B2_DIM), dtype=np.float32)
        role_masks = np.zeros((len(pairs), N_ROLES), dtype=np.float32)
        triple_matrix = np.zeros((len(pairs), TRIPLE_DIM), dtype=np.float32)
        motif_available = []
        context_status = []
        selected_z = []
        ctx_reason = []
        ctx_source = []

        labels = getattr(data, "edge_label", None)
        if labels is not None:
            y = labels.detach().cpu().numpy().astype(float).reshape(-1)
        else:
            y = np.full(len(pairs), np.nan)

        for i, (u, v) in enumerate(pairs):
            key = (str(u), str(v))
            if key in triples_use:
                meta = triples_use[key]
                parts = []
                for r, p in enumerate(meta["pairs"]):
                    pk = (str(p[0]), str(p[1]))
                    parts.append(pair_vectors[pk])
                # Structural role presence (not GHCR support): Z present ⇒ all 3 roles.
                role_masks[i, :] = 1.0
                matrix[i] = np.concatenate(parts)
                motif_available.append(True)
                context_status.append("valid" if not use_matched_z else "matched_shuffled")
                selected_z.append(str(meta["Z"]))
                ctx_reason.append(
                    str(
                        meta.get(
                            "context_selection_reason",
                            meta.get("context_selection_rule", "lexicographic_first"),
                        )
                    )
                )
                rm = reg_meta.get(key, {})
                ctx_source.append(str(rm.get("selector_source", "")))
                if use_triple:
                    tkey = (str(meta["A"]), str(meta["Z"]), str(meta["G"]))
                    tres = triple_cache.get(tkey)
                    if tres is not None:
                        triple_matrix[i] = np.asarray(tres.vector, dtype=np.float32)
            else:
                # No Z: keep AZ/ZG exactly zero; put direct AG in role index 1.
                # Triple block stays exactly zero (no I(A;G) fallback).
                matrix[i, PAIR_DIM : 2 * PAIR_DIM] = pair_vectors[key]
                role_masks[i, 0] = 0.0
                role_masks[i, 1] = 1.0
                role_masks[i, 2] = 0.0
                motif_available.append(False)
                context_status.append("missing")
                selected_z.append("")
                ctx_reason.append("no_registry_context")
                ctx_source.append("")

        data.hcr_features = torch.as_tensor(matrix, dtype=torch.float32, device=device)
        data.hcr_role_masks = torch.as_tensor(role_masks, dtype=torch.float32, device=device)
        data.hcr_triple_features = torch.as_tensor(
            triple_matrix, dtype=torch.float32, device=device
        )
        data.hcr_motif_available = torch.tensor(motif_available, dtype=torch.bool, device=device)
        data.hcr_enabled = True
        data.hcr_variant = variant
        data.hcr_dim = MOTIF_B2_DIM
        data.hcr_triple_dim = TRIPLE_DIM
        data.hcr_panel = "WAVE7C"
        data.hcr_pair_roles = list(PAIR_ROLES)
        data.hcr_triples_hash = triples_hash
        data.hcr_fit_scope = encoder.fit_scope
        data.hcr_patient_train_fingerprint = encoder.patient_train_fingerprint
        data.wave7c_context_status = list(context_status)
        data.wave7c_selected_z = list(selected_z)
        data.wave7c_context_reason = list(ctx_reason)
        data.wave7c_context_source = list(ctx_source)
        data.wave7c_edge_label_audit = y
        data.wave7c_split_name = split_name

    n_trip_sup = int(sum(1 for r in triple_cache.values() if r.supported)) if use_triple else 0
    print(
        f"\nHCR WAVE7C\n  variant: {variant}\n  B2: {B2_VARIANT} → {MOTIF_B2_DIM}D\n"
        f"  C4_matched_z={use_matched_z}  C5_permute={use_permute}  "
        f"T1_triple={use_triple}"
        + (f" (supported {n_trip_sup}/{len(triple_cache)})" if use_triple else "")
        + f"\n  triples: {len(triples_use)}  hash: {triples_hash[:12]}…"
    )
    return encoder
