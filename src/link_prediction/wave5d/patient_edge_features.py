"""Build static + patient-conditioned edge features for Wave 5D."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import torch

from hcr.pair_encoder import HCRPairEncoder
from hcr.structural_context import load_node_metadata
from hcr.variable_specs_v3 import VARIABLE_SPECS
from link_prediction.wave5d.forward_backward_support import (
    completion_from_tables,
    graph_without_candidate,
    precompute_patient_support_tables,
)
from wnerw.types import NEG_INF
from wnerw.wave5c.patient_activity import build_patient_activity, gate_activation
from wnerw.wave5c.patient_energy import PatientEnergyConfig, build_patient_edge_scores


HGT_CKPT = {
    20260721: "outputs/2026-07-31/14-56-55/best_model.pt",
    20260722: "outputs/2026-07-31/14-58-21/best_model.pt",
    20260723: "outputs/2026-07-31/15-00-08/best_model.pt",
    20260724: "outputs/2026-07-31/15-01-43/best_model.pt",
    20260725: "outputs/2026-07-31/15-03-32/best_model.pt",
}


@dataclass
class EdgeFeatureBundle:
    candidate_ids: list[str]
    sources: list[str]
    targets: list[str]
    labels: np.ndarray
    splits: list[str]
    negative_roles: list[str]
    edge_types: list[str]
    hgt_pair: np.ndarray  # [E, 4H]
    hcr24: np.ndarray  # [E, 24]
    support: np.ndarray  # [E]
    uncertainty: np.ndarray  # [E]
    path_completion: np.ndarray  # [P, E]
    gate: np.ndarray  # [P, E]
    patient_ids: list[str]
    patient_splits: list[str]


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p) - np.log(1.0 - p)


def load_hgt_pair_features(
    cfg: Any,
    train_data,
    candidates: pd.DataFrame,
    seed: int,
    root: Path,
    device: torch.device,
) -> np.ndarray:
    from models.TaskA.hetero_gnn import HeteroReconGNN

    ckpt = root / HGT_CKPT[int(seed)]
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    model = HeteroReconGNN(
        cfg,
        train_data,
        in_channels=int(train_data[train_data.node_types[0]].x.size(-1)),
        hidden_channels=int(getattr(cfg.model, "hidden_channels", 64)),
    ).to(device)
    with torch.no_grad():
        _ = model(train_data.to(device))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    with torch.no_grad():
        emb = model.encode_flat(train_data.to(device)).cpu().numpy()

    # Global name → index
    name_to_idx: dict[str, int] = {}
    offset = 0
    for ntype in train_data.node_types:
        names = [str(n) for n in train_data[ntype].node_name]
        for i, n in enumerate(names):
            name_to_idx[n] = offset + i
        offset += len(names)

    rows = []
    h = emb.shape[1]
    for src, tgt in zip(candidates["source"], candidates["target"]):
        i = name_to_idx.get(str(src))
        j = name_to_idx.get(str(tgt))
        if i is None or j is None:
            rows.append(np.zeros(4 * h, dtype=np.float32))
            continue
        zu, zv = emb[i], emb[j]
        rows.append(np.concatenate([zu, zv, zu * zv, np.abs(zu - zv)]).astype(np.float32))
    return np.stack(rows, axis=0)


def build_structural_hcr24(
    candidates: pd.DataFrame,
    train_patients: pd.DataFrame,
    context_map: Mapping[tuple[str, str], tuple[str, ...]],
) -> np.ndarray:
    """AB ⊕ AY ⊕ BY structural latent pairwise (24-d); plain HCR2 padded otherwise."""

    class _Cfg:
        variant = "binary_compact"
        smoothing = 0.5
        output_dim = 8
        unknown_pair_value = 0.0
        unsupported_pair_mode = "zeros"
        force_zero_features = False
        append_supported_mask = False
        features = None

    enc = HCRPairEncoder.from_hydra(_Cfg(), VARIABLE_SPECS)
    pairs = list(zip(candidates["source"].astype(str), candidates["target"].astype(str)))
    extra: list[tuple[str, str]] = []
    triples: dict[tuple[str, str], tuple[str, str, str]] = {}
    for (a, g), ctxs in context_map.items():
        if not ctxs:
            continue
        z = str(ctxs[0])
        y = str(g)
        triples[(str(a), str(g))] = (str(a), y, z)
        extra.extend([(str(a), z), (str(a), y), (z, y)])

    all_pairs = list({(str(u), str(v)) for u, v in pairs + extra})
    enc.fit(train_patient_df=train_patients, candidate_pairs=all_pairs)

    out = np.zeros((len(pairs), 24), dtype=np.float32)
    for i, (u, v) in enumerate(pairs):
        key = (str(u), str(v))
        if key in triples:
            a, y, z = triples[key]
            parts = [
                enc.transform([(a, z)]).cpu().numpy()[0],
                enc.transform([(a, y)]).cpu().numpy()[0],
                enc.transform([(z, y)]).cpu().numpy()[0],
            ]
            out[i] = np.concatenate(parts).astype(np.float32)
        else:
            base = enc.transform([key]).cpu().numpy()[0]
            out[i, :8] = base[:8]
    return out


def _active_sources(
    activity: Mapping[str, float],
    drug_nodes: Sequence[str],
) -> list[str]:
    return [n for n in drug_nodes if float(activity.get(n, 0.0)) >= 0.5]


def _endpoint_nodes(meta: Mapping[str, Mapping[str, Any]] | pd.DataFrame) -> list[str]:
    if isinstance(meta, pd.DataFrame):
        col = "node" if "node" in meta.columns else "node_id"
        tcol = "node_type" if "node_type" in meta.columns else "type"
        return [
            str(r[col])
            for _, r in meta.iterrows()
            if str(r[tcol]) == "clinical_endpoint"
        ]
    return [
        str(n)
        for n, info in meta.items()
        if str(info.get("node_type", info.get("type", ""))) == "clinical_endpoint"
    ]


def _drug_nodes(meta) -> list[str]:
    if isinstance(meta, pd.DataFrame):
        col = "node" if "node" in meta.columns else "node_id"
        tcol = "node_type" if "node_type" in meta.columns else "type"
        return [
            str(r[col])
            for _, r in meta.iterrows()
            if str(r[tcol]) == "drug_exposure"
        ]
    return [
        str(n)
        for n, info in meta.items()
        if str(info.get("node_type", info.get("type", ""))) == "drug_exposure"
    ]


def compute_patient_matrices(
    *,
    g_train: nx.DiGraph,
    evidence: pd.DataFrame,
    candidates: pd.DataFrame,
    patients: pd.DataFrame,
    context_map: Mapping[tuple[str, str], tuple[str, ...]],
    admissible: set[str],
    meta,
    energy_cfg: PatientEnergyConfig,
    max_patients_per_split: int = 64,
    seed: int = 0,
    random_path_weights: bool = False,
    context_column_map: Mapping[str, str] | None = None,
    leave_one_out: bool = True,
    patient_id_permutation: Mapping[str, str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Return path_completion [P,E], gate [P,E], patient_ids, patient_splits."""
    rng = np.random.default_rng(seed)
    selected_rows = []
    for split in ("train", "valid", "test"):
        part = patients[patients["split"].astype(str).str.lower() == split]
        if part.empty:
            continue
        take = min(max_patients_per_split, len(part))
        idx = rng.choice(part.index.to_numpy(), size=take, replace=False)
        selected_rows.append(patients.loc[idx])
    if not selected_rows:
        raise RuntimeError("No patients selected for Wave 5D features.")
    pdf = pd.concat(selected_rows, axis=0)

    drugs = _drug_nodes(meta)
    endpoints = _endpoint_nodes(meta)
    edges = list(zip(candidates["source"].astype(str), candidates["target"].astype(str)))
    n_e = len(edges)
    patient_ids: list[str] = []
    patient_splits: list[str] = []
    path_rows: list[np.ndarray] = []
    gate_rows: list[np.ndarray] = []

    cmap = dict(context_map)
    colmap_tuple = tuple((k, v) for k, v in (context_column_map or {}).items())

    train_edge_set = {
        (str(r.source), str(r.target))
        for r in evidence.itertuples(index=False)
        if bool(getattr(r, "edge_in_train", False))
    }

    for _, prow in pdf.iterrows():
        pid = str(prow["patient_id"])
        use_pid = str(patient_id_permutation.get(pid, pid)) if patient_id_permutation else pid
        if patient_id_permutation is not None:
            # Use another patient's row values
            alt = patients[patients["patient_id"].astype(str) == use_pid]
            row = alt.iloc[0] if len(alt) else prow
        else:
            row = prow

        activity = build_patient_activity(row, admissible)
        cfg = PatientEnergyConfig(
            temperature=energy_cfg.temperature,
            length_penalty=energy_cfg.length_penalty,
            node_activity_weight=energy_cfg.node_activity_weight,
            gate_weight=energy_cfg.gate_weight,
            static_hcr_weight=energy_cfg.static_hcr_weight,
            hcr_probability_column=energy_cfg.hcr_probability_column,
            context_column_map=colmap_tuple,
        )
        scores = build_patient_edge_scores(
            graph=g_train,
            edge_evidence=evidence,
            patient_activity=activity,
            admissible_activity_nodes=admissible,
            context_map=cmap,
            config=cfg,
        )
        if random_path_weights:
            scores = {
                e: float(rng.normal(loc=-0.2, scale=0.5))
                for e in scores
            }

        sources = _active_sources(activity, drugs)
        if not sources:
            # fallback: any drug column ≥ threshold even if not in admissible
            sources = [
                d
                for d in drugs
                if d in row.index and float(pd.to_numeric(row[d], errors="coerce") or 0) >= 0.5
            ]

        log_f, log_b = precompute_patient_support_tables(
            g_train, scores, sources, endpoints
        )

        path_vec = np.full(n_e, NEG_INF, dtype=np.float64)
        gate_vec = np.zeros(n_e, dtype=np.float64)

        for j, (u, v) in enumerate(edges):
            # Gate activation for structural contexts on (u,v)
            contexts = cmap.get((u, v), ())
            if contexts:
                gate_vec[j] = gate_activation(u, contexts, activity)

            if leave_one_out and (u, v) in train_edge_set and g_train.has_edge(u, v):
                g_lo = graph_without_candidate(g_train, (u, v))
                lf, lb = precompute_patient_support_tables(
                    g_lo, scores, sources, endpoints
                )
                path_vec[j] = completion_from_tables(
                    (u, v), lf, lb, sources, endpoints
                )
            else:
                path_vec[j] = completion_from_tables(
                    (u, v), log_f, log_b, sources, endpoints
                )

        # Replace -inf with finite floor for MLP stability
        finite = path_vec[np.isfinite(path_vec)]
        floor = float(finite.min() - 1.0) if len(finite) else -20.0
        path_vec = np.where(np.isfinite(path_vec), path_vec, floor)
        # Standardize lightly per patient
        mu, sd = float(path_vec.mean()), float(path_vec.std() + 1e-6)
        path_vec = (path_vec - mu) / sd

        path_rows.append(path_vec.astype(np.float32))
        gate_rows.append(gate_vec.astype(np.float32))
        patient_ids.append(pid)
        patient_splits.append(str(prow["split"]).lower())

    return (
        np.stack(path_rows, axis=0),
        np.stack(gate_rows, axis=0),
        patient_ids,
        patient_splits,
    )


def build_edge_feature_bundle(
    *,
    cfg: Any,
    train_data,
    evidence: pd.DataFrame,
    candidates: pd.DataFrame,
    patients: pd.DataFrame,
    context_map: Mapping[tuple[str, str], tuple[str, ...]],
    admissible: set[str],
    g_train: nx.DiGraph,
    seed: int,
    root: Path,
    energy_cfg: PatientEnergyConfig,
    max_patients_per_split: int = 64,
    device: torch.device | None = None,
    control: str = "none",
) -> EdgeFeatureBundle:
    device = device or torch.device("cpu")
    from data.patient_matrix import train_patient_df

    train_patients = train_patient_df(patients)
    hgt_pair = load_hgt_pair_features(cfg, train_data, candidates, seed, root, device)
    hcr24 = build_structural_hcr24(candidates, train_patients, context_map)

    ev = evidence.drop_duplicates("candidate_id").set_index("candidate_id")
    aligned = ev.reindex(candidates["candidate_id"].astype(str))
    if "hcr_supported" in aligned.columns:
        supp = aligned["hcr_supported"].fillna(False).astype(float).to_numpy(dtype=np.float32)
    else:
        supp = np.zeros(len(candidates), dtype=np.float32)
    if "hcr_uncertainty" in aligned.columns:
        uncertainty = (
            aligned["hcr_uncertainty"].fillna(0.5).to_numpy(dtype=np.float32)
        )
    else:
        uncertainty = np.full(len(candidates), 0.5, dtype=np.float32)

    # Controls
    random_path = control == "random_path_weights"
    leave_one = control != "no_leave_one_out"
    colmap = None
    perm = None
    cmap = dict(context_map)

    if control == "patient_shuffle":
        rng = np.random.default_rng(seed + 17)
        ids = patients["patient_id"].astype(str).tolist()
        shuffled = rng.permutation(ids)
        perm = dict(zip(ids, shuffled))
    elif control == "context_shuffle":
        rng = np.random.default_rng(seed + 19)
        ctx_nodes = sorted({c for ctx in cmap.values() for c in ctx})
        if ctx_nodes:
            alt = list(rng.permutation(ctx_nodes))
            colmap = dict(zip(ctx_nodes, alt))
    elif control == "matched_random_context":
        rng = np.random.default_rng(seed + 23)
        pool = sorted(admissible)
        new_map = {}
        for (u, v), ctxs in cmap.items():
            if not ctxs or not pool:
                new_map[(u, v)] = ctxs
                continue
            # Prevalence-agnostic matched substitute: random admissible non-parent.
            choices = [x for x in pool if x not in ctxs and x != u]
            if not choices:
                new_map[(u, v)] = ctxs
            else:
                new_map[(u, v)] = (str(rng.choice(choices)),)
        cmap = new_map

    path_c, gate, pids, psplits = compute_patient_matrices(
        g_train=g_train,
        evidence=evidence,
        candidates=candidates,
        patients=patients,
        context_map=cmap,
        admissible=admissible,
        meta=load_node_metadata(cfg),
        energy_cfg=energy_cfg,
        max_patients_per_split=max_patients_per_split,
        seed=seed,
        random_path_weights=random_path,
        context_column_map=colmap,
        leave_one_out=leave_one,
        patient_id_permutation=perm,
    )

    return EdgeFeatureBundle(
        candidate_ids=candidates["candidate_id"].astype(str).tolist(),
        sources=candidates["source"].astype(str).tolist(),
        targets=candidates["target"].astype(str).tolist(),
        labels=candidates["label"].to_numpy(dtype=np.float32),
        splits=candidates["mask_split"].astype(str).tolist(),
        negative_roles=candidates["negative_role"].astype(str).tolist(),
        edge_types=candidates["edge_type"].astype(str).tolist(),
        hgt_pair=hgt_pair.astype(np.float32),
        hcr24=hcr24.astype(np.float32),
        support=supp,
        uncertainty=uncertainty,
        path_completion=path_c,
        gate=gate,
        patient_ids=pids,
        patient_splits=psplits,
    )


def save_feature_bundle(bundle: EdgeFeatureBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        candidate_ids=np.array(bundle.candidate_ids),
        sources=np.array(bundle.sources),
        targets=np.array(bundle.targets),
        labels=bundle.labels,
        splits=np.array(bundle.splits),
        negative_roles=np.array(bundle.negative_roles),
        edge_types=np.array(bundle.edge_types),
        hgt_pair=bundle.hgt_pair,
        hcr24=bundle.hcr24,
        support=bundle.support,
        uncertainty=bundle.uncertainty,
        path_completion=bundle.path_completion,
        gate=bundle.gate,
        patient_ids=np.array(bundle.patient_ids),
        patient_splits=np.array(bundle.patient_splits),
    )


def load_feature_bundle(path: Path) -> EdgeFeatureBundle:
    z = np.load(path, allow_pickle=True)
    return EdgeFeatureBundle(
        candidate_ids=z["candidate_ids"].tolist(),
        sources=z["sources"].tolist(),
        targets=z["targets"].tolist(),
        labels=z["labels"],
        splits=z["splits"].tolist(),
        negative_roles=z["negative_roles"].tolist(),
        edge_types=z["edge_types"].tolist(),
        hgt_pair=z["hgt_pair"],
        hcr24=z["hcr24"],
        support=z["support"],
        uncertainty=z["uncertainty"],
        path_completion=z["path_completion"],
        gate=z["gate"],
        patient_ids=z["patient_ids"].tolist(),
        patient_splits=z["patient_splits"].tolist(),
    )
