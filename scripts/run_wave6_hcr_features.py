#!/usr/bin/env python3
"""Wave 6 / Stage 4 pilot+full: mixed-type HCR features (no GNN training)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from hcr.variable_specs_v3 import VARIABLE_SPECS
from hcr_mixed.compare_dependence import abs_spearman, normalized_hsic
from hcr_mixed.core import ConditioningSelector, infer_kind, mixed_feature_8d

ROOT = Path(__file__).resolve().parents[1]


def find_col(df: pd.DataFrame, names: list[str]) -> str:
    for name in names:
        if name in df.columns:
            return name
    raise KeyError(f"Missing one of columns: {names}")


def resolve_kind(name: str, train: pd.DataFrame) -> str:
    if name in VARIABLE_SPECS:
        return str(VARIABLE_SPECS[name].variable_type.value)
    return infer_kind(train[name], None)


def _parse_z_list(value) -> list:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        return json.loads(value) if value.strip() else []
    return list(value)


def select_with_reason(selector: ConditioningSelector, u: str, v: str, mandatory, forbidden):
    mandatory_z = [
        z
        for z in dict.fromkeys(mandatory)
        if z in selector.train_samples.columns and z not in forbidden
    ]
    all_z = selector.select(u, v, mandatory=mandatory, forbidden=forbidden)
    selected_z = [z for z in all_z if z not in mandatory_z]
    if mandatory_z and selected_z:
        reason = "mandatory_plus_safe_common_ancestor"
    elif mandatory_z:
        reason = "mandatory_only"
    elif selected_z:
        reason = "safe_common_ancestor_only"
    else:
        reason = "empty_z"
    return mandatory_z, selected_z, all_z, reason


def motif_z(selector: ConditioningSelector, a: str, g: str, b: str, y: str):
    return {
        "AB": select_with_reason(selector, a, b, (), {g, y}),
        "AY": select_with_reason(selector, a, y, (b,), {g}),
        "BY": select_with_reason(selector, b, y, (a,), {g}),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=os.environ.get("PHARMA_DATA_ROOT"))
    p.add_argument("--splits", default=None)
    p.add_argument("--context-registry", required=True)
    p.add_argument("--scenario", default="clean")
    p.add_argument("--degree", type=int, default=4)
    p.add_argument("--max-z", type=int, default=2)
    p.add_argument("--ridge-alpha", type=float, default=0.01)
    p.add_argument("--bootstrap", type=int, default=10)
    p.add_argument("--hsic-max-n", type=int, default=1000)
    p.add_argument("--out-dir", default="outputs/wave6_hcr/pilot")
    p.add_argument(
        "--frozen-conditioning",
        default=None,
        help="Use fixed Z names from FROZEN_CONDITIONING_REGISTRY.csv (refit HCR values only).",
    )
    p.add_argument(
        "--write-frozen-registry",
        action="store_true",
        help="Also write FROZEN_CONDITIONING_REGISTRY.csv (Etap 2 freeze).",
    )
    args = p.parse_args()
    if not args.data_root:
        raise SystemExit("Set --data-root or PHARMA_DATA_ROOT")

    root = Path(args.data_root)
    splits_path = Path(args.splits) if args.splits else root.parent / "splits" / "patient_splits_v3.csv"
    if not splits_path.exists():
        splits_path = root / "patient_splits_v3.csv"

    sample_name = f"synthetic_pharmacotherapy_v3_samples_{args.scenario}.csv"
    nodes = pd.read_csv(root / "synthetic_pharmacotherapy_v3_nodes.csv")
    edges = pd.read_csv(root / "synthetic_pharmacotherapy_v3_edges_audited.csv")
    samples = pd.read_csv(root / sample_name)
    splits = pd.read_csv(splits_path)
    contexts = pd.read_csv(args.context_registry)

    sid_s = find_col(samples, ["patient_id", "patient", "id"])
    sid_p = find_col(splits, ["patient_id", "patient", "id"])
    split_col = find_col(splits, ["split", "partition", "set"])
    train_ids = set(splits.loc[splits[split_col].astype(str).str.lower().eq("train"), sid_p].astype(str))
    train = samples[samples[sid_s].astype(str).isin(train_ids)].reset_index(drop=True)

    source_col = find_col(contexts, ["source", "source_a", "candidate_source"])
    target_col = find_col(contexts, ["target", "target_g", "candidate_target"])
    b_col = find_col(contexts, ["co_parent", "context_b", "parent_b", "B"])
    y_col = find_col(contexts, ["downstream", "downstream_y", "child_y", "Y"])
    candidate_col = find_col(contexts, ["candidate_id"])

    value_types = {c: resolve_kind(c, train) for c in train.columns if c != sid_s}
    selector = ConditioningSelector(nodes, edges, train, max_total_z=args.max_z)

    frozen_lookup = {}
    if args.frozen_conditioning:
        frozen = pd.read_csv(args.frozen_conditioning)
        for _, fr in frozen.iterrows():
            key = (str(fr["candidate_id"]), str(fr["pair_role"]))
            frozen_lookup[key] = {
                "u": str(fr["u"]),
                "v": str(fr["v"]),
                "mandatory_z": _parse_z_list(fr["mandatory_z"]),
                "selected_z": _parse_z_list(fr["selected_z"]),
                "all_z": _parse_z_list(fr["all_z"]),
                "selection_reason": str(fr.get("selection_reason", "frozen")),
            }

    feat_rows, cond_rows = [], []
    for _, row in contexts.iterrows():
        cid = str(row[candidate_col])
        a, g, b, y = map(str, [row[source_col], row[target_col], row[b_col], row[y_col]])
        zinfo = None if frozen_lookup else motif_z(selector, a, g, b, y)
        for role, (u, v) in {"AB": (a, b), "AY": (a, y), "BY": (b, y)}.items():
            if frozen_lookup:
                key = (cid, role)
                if key not in frozen_lookup:
                    raise SystemExit(f"Missing frozen Z for {cid} {role}")
                fz = frozen_lookup[key]
                u, v = fz["u"], fz["v"]
                mandatory_z = fz["mandatory_z"]
                selected_z = fz["selected_z"]
                all_z = fz["all_z"]
                reason = fz["selection_reason"]
            else:
                mandatory_z, selected_z, all_z, reason = zinfo[role]
            cond_rows.append(
                {
                    "candidate_id": cid,
                    "pair_role": role,
                    "u": u,
                    "v": v,
                    "gate": g,
                    "mandatory_z": json.dumps(mandatory_z),
                    "selected_z": json.dumps(selected_z),
                    "all_z": json.dumps(all_z),
                    "selection_reason": reason,
                }
            )
            if u not in train.columns or v not in train.columns:
                feat_rows.append(
                    {
                        "candidate_id": cid,
                        "pair_role": role,
                        "u": u,
                        "v": v,
                        "kind_u": "missing",
                        "kind_v": "missing",
                        "route": "unsupported",
                        "support": 0.0,
                    }
                )
                continue
            ku, kv = value_types[u], value_types[v]
            spear = abs_spearman(train[u], train[v])
            hsic = normalized_hsic(train[u], train[v], max_n=args.hsic_max_n)
            if ku == "binary" and kv == "binary":
                feat_rows.append(
                    {
                        "candidate_id": cid,
                        "pair_role": role,
                        "u": u,
                        "v": v,
                        "kind_u": ku,
                        "kind_v": kv,
                        "route": "existing_binary_compact_8d",
                        "raw_energy": np.nan,
                        "conditional_energy": np.nan,
                        "support": float((train[u].notna() & train[v].notna()).mean()),
                        "bootstrap_sd": np.nan,
                        "abs_spearman": spear,
                        "nHSIC": hsic,
                        "z": json.dumps(all_z),
                    }
                )
                continue
            zcols = [z for z in all_z if z in train.columns]
            zdf = train[zcols] if zcols else None
            zkinds = {z: value_types[z] for z in zcols}
            feature = mixed_feature_8d(
                train[u],
                train[v],
                ku,
                kv,
                zdf,
                zkinds,
                degree=args.degree,
                bootstrap_repeats=args.bootstrap,
            )
            feat_rows.append(
                {
                    "candidate_id": cid,
                    "pair_role": role,
                    "u": u,
                    "v": v,
                    "kind_u": ku,
                    "kind_v": kv,
                    "route": "jarek_mixed_hcr_8d",
                    "hcr_0": float(feature[0]),
                    "hcr_1": float(feature[1]),
                    "hcr_2": float(feature[2]),
                    "hcr_3": float(feature[3]),
                    "raw_energy": float(feature[4]),
                    "conditional_energy": float(feature[5]),
                    "support": float(feature[6]),
                    "bootstrap_sd": float(feature[7]),
                    "abs_spearman": spear,
                    "nHSIC": hsic,
                    "z": json.dumps(zcols),
                }
            )

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    feat = pd.DataFrame(feat_rows)
    cond = pd.DataFrame(cond_rows)
    feat_path = out_dir / "mixed_hcr_pair_features.csv"
    cond_path = out_dir / "conditioning_registry.csv"
    feat.to_csv(feat_path, index=False)
    cond.to_csv(cond_path, index=False)

    frozen_path = None
    if args.write_frozen_registry:
        frozen_cols = [
            "candidate_id",
            "pair_role",
            "u",
            "v",
            "mandatory_z",
            "selected_z",
            "all_z",
            "selection_reason",
        ]
        frozen_path = out_dir / "FROZEN_CONDITIONING_REGISTRY.csv"
        cond[frozen_cols].to_csv(frozen_path, index=False)
        # canonical copy at wave6 root
        (ROOT / "outputs/wave6_hcr/FROZEN_CONDITIONING_REGISTRY.csv").write_text(
            frozen_path.read_text()
        )

    status = {
        "wave": "WAVE6_MIXED_HCR",
        "stage": (
            "feature_full"
            if args.write_frozen_registry or args.frozen_conditioning
            else "feature_pilot_or_full"
        ),
        "scenario": args.scenario,
        "n_motifs": int(contexts[candidate_col].nunique()),
        "n_pair_rows": int(len(feat)),
        "degree": args.degree,
        "max_z": args.max_z,
        "ridge_alpha": args.ridge_alpha,
        "bootstrap": args.bootstrap,
        "hsic_max_n": args.hsic_max_n,
        "context_registry": str(Path(args.context_registry).resolve()),
        "frozen_conditioning": (
            str(Path(args.frozen_conditioning).resolve()) if args.frozen_conditioning else None
        ),
        "data_root": str(root.resolve()),
        "route_counts": feat["route"].value_counts(dropna=False).to_dict(),
        "note": (
            "No GNN training. Binary-binary uses existing_binary_compact_8d route marker only. "
            "Z fixed when --frozen-conditioning is set."
        ),
    }
    (out_dir / "WAVE6_FEATURE_STATUS.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))
    print("Saved:", feat_path)
    print("Saved:", cond_path)
    if frozen_path is not None:
        print("Saved:", frozen_path)
        print("Saved:", ROOT / "outputs/wave6_hcr/FROZEN_CONDITIONING_REGISTRY.csv")


if __name__ == "__main__":
    main()
