#!/usr/bin/env python3
"""Attach-only V1 benchmark + regression vs reference bootstrap loop."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from hcr.variable_spec import VariableType
from hcr.variable_specs_v3 import VARIABLE_SPECS
from hcr.wave7.encoder import Wave7HCRConfig, Wave7PairEncoder, _a11_from_binary_sample
from hcr.wave7.motif import build_candidate_coparent_triples
from hcr.wave7.v1_bootstrap import (
    _bootstrap_sd_a11_reference,
    _bootstrap_sd_a11_vectorized,
    v1_bootstrap_seed,
)
from wnerw.wave5c.edge_context_registry import build_context_map
from data.patient_matrix import load_patient_matrix_with_split, train_patient_df


def _make_cfg():
    import os
    from types import SimpleNamespace

    root = Path(os.environ.get("GSN_PROJECT_ROOT", Path.home() / "Desktop/GSN Graphs dysertation 2026"))
    data_root = root / "2 v3. Data" / "dataset_v3"
    return SimpleNamespace(
        data=SimpleNamespace(
            candidate_seed=20260722,
            dataset=SimpleNamespace(
                scenario="clean",
                root_dir=str(data_root),
                patient_split_file="patient_splits_v3.csv",
                nodes_file="synthetic_pharmacotherapy_v3_nodes.csv",
                samples={
                    "clean": "synthetic_pharmacotherapy_v3_samples_clean.csv",
                    "hidden_confounder": "synthetic_pharmacotherapy_v3_samples_hidden_confounder.csv",
                    "selection_bias": "synthetic_pharmacotherapy_v3_samples_selection_bias.csv",
                    "no_overlap": "synthetic_pharmacotherapy_v3_samples_no_overlap.csv",
                    "noisy_documentation": "synthetic_pharmacotherapy_v3_samples_noisy_documentation.csv",
                    "multihospital": "synthetic_pharmacotherapy_v3_samples_multihospital.csv",
                },
            ),
        ),
        training=SimpleNamespace(seed=20260722),
    )


def main() -> int:
    out_dir = ROOT / "outputs" / "wave7" / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Synthetic old-path timing proxy: reference bootstrap loop on real pairs.
    # New path: vectorized + pair cache on the same pairs.
    cfg = _make_cfg()
    patient_df = load_patient_matrix_with_split(cfg)
    train = train_patient_df(patient_df)

    reg = pd.read_csv(ROOT / "outputs/wave5c/registry/edge_context_registry.csv")
    reg = reg[reg["seed"] == 20260722]
    triples = build_candidate_coparent_triples(build_context_map(reg))

    binary_names = {
        n
        for n, spec in VARIABLE_SPECS.items()
        if spec.variable_type == VariableType.BINARY
    }
    unique_pairs = sorted(
        {
            tuple(sorted((str(a), str(b))))
            for meta in triples.values()
            for a, b in meta["pairs"]
            if a in binary_names and b in binary_names
        }
    )[:80]
    assert len(unique_pairs) >= 20

    # Build encoder for column mapping
    hcfg = Wave7HCRConfig(
        variant="W7_V1_GHCR_BINARY_ONEHOT",
        bootstrap_repeats=30,
        v1_smoothing=0.0,
        bootstrap_seed=20260722,
    )
    enc = Wave7PairEncoder(hcfg)
    enc.scenario = "clean"
    enc.fit(train, {n for uv in unique_pairs for n in uv})

    # Regression on ≥20 pairs
    max_abs_diff = 0.0
    regression_ok = True
    for u_name, v_name in unique_pairs[:25]:
        col_u, col_v = enc._column(u_name), enc._column(v_name)
        u = pd.to_numeric(train[col_u], errors="coerce").to_numpy(dtype=np.float64)
        v = pd.to_numeric(train[col_v], errors="coerce").to_numpy(dtype=np.float64)
        complete = np.isfinite(u) & np.isfinite(v)
        uu = np.clip(np.round(u[complete]), 0, 1)
        vv = np.clip(np.round(v[complete]), 0, 1)
        if len(uu) < 50:
            continue
        a11 = _a11_from_binary_sample(uu, vv, smoothing=0.0)
        seed = v1_bootstrap_seed(
            global_seed=20260722, scenario="clean", u_name=u_name, v_name=v_name
        )
        sd_ref, _ = _bootstrap_sd_a11_reference(
            uu, vv, repeats=30, smoothing=0.0, seed=seed
        )
        sd_vec, _ = _bootstrap_sd_a11_vectorized(
            uu, vv, repeats=30, smoothing=0.0, seed=seed
        )
        vec, meta = enc.transform_pair(u_name, v_name)
        diffs = [
            abs(float(meta["a11"]) - a11),
            abs(float(meta["energy"]) - a11**2),
            abs(float(meta["support"]) - (len(uu) / len(train))),
            abs(float(meta["bootstrap_sd_a11"]) - sd_ref),
            abs(sd_vec - sd_ref),
        ]
        max_abs_diff = max(max_abs_diff, max(diffs))
        if (
            diffs[0] > 1e-12
            or diffs[1] > 1e-12
            or diffs[2] > 1e-12
            or diffs[3] > 1e-9
            or diffs[4] > 1e-9
        ):
            regression_ok = False
            break

    # Timing: realistic attach pressure.
    # Old: recompute every motif-pair lookup (unique × 3 splits).
    # New: compute each unique pair once, then cache hits for reuse.
    reuse_factor = 3  # train/valid/test
    enc_old_t = 0.0
    t0 = time.perf_counter()
    for _ in range(reuse_factor):
        for u_name, v_name in unique_pairs:
            col_u, col_v = enc._column(u_name), enc._column(v_name)
            u = pd.to_numeric(train[col_u], errors="coerce").to_numpy(dtype=np.float64)
            v = pd.to_numeric(train[col_v], errors="coerce").to_numpy(dtype=np.float64)
            complete = np.isfinite(u) & np.isfinite(v)
            uu = np.clip(np.round(u[complete]), 0, 1)
            vv = np.clip(np.round(v[complete]), 0, 1)
            if len(uu) < 50:
                continue
            seed = v1_bootstrap_seed(
                global_seed=20260722, scenario="clean", u_name=u_name, v_name=v_name
            )
            _bootstrap_sd_a11_reference(uu, vv, repeats=30, smoothing=0.0, seed=seed)
            _a11_from_binary_sample(uu, vv, smoothing=0.0)
    enc_old_t = time.perf_counter() - t0

    enc2 = Wave7PairEncoder(hcfg)
    enc2.scenario = "clean"
    enc2.fit(train, {n for uv in unique_pairs for n in uv})
    t0 = time.perf_counter()
    for u_name, v_name in unique_pairs:
        enc2.transform_pair(u_name, v_name)
    for _ in range(reuse_factor - 1):
        for u_name, v_name in unique_pairs:
            enc2.transform_pair(u_name, v_name)
    enc_new_t = time.perf_counter() - t0

    speedup = (enc_old_t / enc_new_t) if enc_new_t > 0 else float("inf")

    # Compare to legacy V1 clean cache (bootstrap seed formula changed → likely fail).
    legacy_cache = ROOT / "outputs/hcr/wave7/W7_V1_GHCR_BINARY_ONEHOT__clean.csv"
    legacy_compatible = False
    legacy_max_diff = None
    if legacy_cache.exists() and regression_ok:
        legacy = pd.read_csv(legacy_cache)
        # Rebuild a few motif rows is heavy; compare unique pair vectors via re-attach
        # of first-block features for candidates where mask==1 on all three — skip.
        # Instead: mark based on seed-policy note.
        legacy_compatible = False
        legacy_max_diff = "seed_formula_changed_bootstrap_sd"

    payload = {
        "old_runtime_seconds": enc_old_t,
        "new_runtime_seconds": enc_new_t,
        "speedup": speedup,
        "bootstrap_repeats": 30,
        "unique_pairs": len(unique_pairs),
        "cache_hits": enc2.cache_hits,
        "cache_misses": enc2.cache_misses,
        "maximum_absolute_vector_difference": max_abs_diff,
        "patient_train_fingerprint": enc.patient_train_fingerprint,
        "regression_passed": regression_ok,
        "legacy_v1_clean_compatible": legacy_compatible,
        "legacy_note": legacy_max_diff,
        "fit_scope": "patient_train_only",
    }
    out_path = out_dir / "V1_ATTACH_BENCHMARK.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))

    decision = {
        "v1_clean_status": "legacy_invalid_for_controlled_wave7"
        if not legacy_compatible
        else "keep",
        "reason": (
            "Optimized V1 uses deterministic canonical-pair bootstrap seeds; "
            "pre-optimization V1 clean used ordered hash seeds. "
            "Do not mix both in the primary result table — rerun V1 clean."
        ),
        "regression_passed": regression_ok,
        "benchmark": str(out_path),
    }
    (out_dir / "V1_CLEAN_RESUME_POLICY.json").write_text(json.dumps(decision, indent=2))
    print("Wrote", out_path)
    print("Wrote", out_dir / "V1_CLEAN_RESUME_POLICY.json")
    return 0 if regression_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
