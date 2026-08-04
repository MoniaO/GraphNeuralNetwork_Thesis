#!/usr/bin/env python3
"""Export Task A wave-2 architecture runs into dated, clearly labeled folders.

Stages
------
ARCH_SCREENING       → outputs/wave2_architecture_YYYY-MM-DD/
ARCH_DEPTH           → outputs/wave2_architecture_YYYY-MM-DD_DEPTH/
ARCH_HEADS           → outputs/wave2_architecture_YYYY-MM-DD_HEADS/
ARCH_HEADS_BOUNDARY  → outputs/wave2_architecture_YYYY-MM-DD_HEADS_BOUNDARY/
ARCH_FINAL           → outputs/wave2_architecture_YYYY-MM-DD_FINAL/

Wave 1 baselines/ablations stay in:
  outputs/wave1_baselines_ablations/
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WANDB = PROJECT_ROOT / "wandb"

STAGE_TO_DIR_SUFFIX = {
    "ARCH_SCREENING": "",
    "ARCH_DEPTH": "_DEPTH",
    "ARCH_HEADS": "_HEADS",
    "ARCH_HEADS_BOUNDARY": "_HEADS_BOUNDARY",
    "ARCH_FINAL": "_FINAL",
}


def _pick(summary: dict, *keys, default=None):
    for key in keys:
        if key in summary and summary[key] is not None:
            return summary[key]
    return default


def _cfg_text(run_dir: Path) -> str:
    path = run_dir / "files" / "config.yaml"
    return path.read_text(errors="ignore") if path.exists() else ""


def _meta_args(run_dir: Path) -> str:
    path = run_dir / "files" / "wandb-metadata.json"
    if not path.exists():
        return ""
    try:
        meta = json.loads(path.read_text())
    except Exception:
        return ""
    args = meta.get("args") or []
    return " ".join(args) if isinstance(args, list) else str(args)


def _normalize_model(model: str, blob: str) -> str:
    model = (model or "").lower().strip()
    if not model:
        for cand in ("hetero_sage_matched", "hetero_gatv2", "hgt"):
            if cand in blob:
                return cand
    if model in {"gatv2"}:
        return "hetero_gatv2"
    if model in {"sage_matched"}:
        return "hetero_sage_matched"
    return model


def harvest_rows(stage: str) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(WANDB.glob("run-*")):
        summary_path = run_dir / "files" / "wandb-summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        blob = _cfg_text(run_dir) + "\n" + _meta_args(run_dir) + "\n" + json.dumps(summary)

        wave = str(_pick(summary, "experiment_name", default="") or "")
        if wave != stage and stage not in blob:
            continue

        model = _normalize_model(
            str(_pick(summary, "encoder_name", default="") or ""),
            blob,
        )
        seed = _pick(summary, "training_seed")
        if seed is None:
            m = re.search(r"training\.seed=(\d+)", blob)
            seed = int(m.group(1)) if m else None

        num_layers = _pick(summary, "num_layers")
        if num_layers is None:
            m = re.search(r"num_layers[=:]\s*(\d+)", blob)
            num_layers = int(m.group(1)) if m else None

        heads = _pick(summary, "heads")
        if heads is None:
            m = re.search(r"\bheads[=:]\s*(\d+)", blob)
            heads = int(m.group(1)) if m else None

        rows.append(
            {
                "wave": "wave2_architecture",
                "stage": stage,
                "experiment_name": stage,
                "model": model,
                "num_layers": int(num_layers) if num_layers is not None else None,
                "hidden_channels": _pick(summary, "hidden_channels"),
                "heads": int(heads) if heads is not None else None,
                "relation_aggr": _pick(summary, "relation_aggr"),
                "training_seed": int(seed) if seed is not None else None,
                "scenario": "clean",
                "feature_profile": "empirical",
                "candidate_fingerprint": _pick(summary, "candidate_fingerprint"),
                "feature_fingerprint": _pick(summary, "feature_fingerprint"),
                "graph_fingerprint": _pick(summary, "graph_fingerprint"),
                "parameter_count_total": _pick(summary, "parameter_count_total"),
                "parameter_count_trainable": _pick(
                    summary, "parameter_count_trainable"
                ),
                "best_epoch": _pick(summary, "best_epoch"),
                "valid_auprc": _pick(
                    summary, "valid_auprc", "best_valid_AUPRC", "best_valid_metric"
                ),
                "valid_auroc": _pick(summary, "valid_auc", "valid_auroc"),
                "valid_brier": _pick(summary, "valid_brier"),
                "valid_f2": _pick(summary, "valid_f2"),
                "test_auprc": _pick(summary, "test_auprc"),
                "test_auroc": _pick(summary, "test_auc", "test_auroc"),
                "test_brier": _pick(summary, "test_brier"),
                "test_f2": _pick(summary, "test_f2"),
                "run_id": run_dir.name.split("-")[-1],
                "run_dir": run_dir.name,
                "mtime": run_dir.stat().st_mtime,
            }
        )

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows).dropna(subset=["model", "training_seed"])
    frame = frame.sort_values("mtime")

    dedupe_keys = ["model", "training_seed", "num_layers"]
    if stage in {"ARCH_HEADS", "ARCH_HEADS_BOUNDARY", "ARCH_FINAL"}:
        dedupe_keys.append("heads")
    frame = frame.drop_duplicates(subset=dedupe_keys, keep="first")
    return frame.sort_values(
        ["model", "num_layers", "heads", "training_seed"],
        na_position="last",
    ).reset_index(drop=True)


def write_stage_exports(stage: str, day: str | None = None) -> Path:
    day = day or date.today().isoformat()
    suffix = STAGE_TO_DIR_SUFFIX[stage]
    out_dir = PROJECT_ROOT / "outputs" / f"wave2_architecture_{day}{suffix}"
    by_model = out_dir / "by_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    by_model.mkdir(parents=True, exist_ok=True)

    frame = harvest_rows(stage)
    all_path = out_dir / f"{day}_wave2_{stage}_all_runs.csv"
    summary_path = out_dir / f"{day}_wave2_{stage}_summary.csv"
    manifest_path = out_dir / "MANIFEST.txt"

    if frame.empty:
        manifest_path.write_text(
            "\n".join(
                [
                    f"Wave: 2 — architecture / stage={stage}",
                    f"Date folder: {day}{suffix}",
                    "Status: no matching local wandb runs yet.",
                    f"Re-export: python scripts/export_wave2_architecture_results.py --stage {stage}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"No {stage} runs yet — wrote {manifest_path}")
        return out_dir

    frame.to_csv(all_path, index=False)

    for model_name, group in frame.groupby("model"):
        safe = str(model_name).replace("/", "_")
        group.to_csv(
            by_model / f"{day}_wave2_{stage}_{safe}_runs.csv",
            index=False,
        )

    group_cols = ["model", "num_layers"]
    if "heads" in frame.columns and frame["heads"].notna().any():
        group_cols.append("heads")

    summary = (
        frame.groupby(group_cols, dropna=False)
        .agg(
            n_seeds=("training_seed", "nunique"),
            mean_valid_auprc=("valid_auprc", "mean"),
            std_valid_auprc=("valid_auprc", "std"),
            mean_test_auprc=("test_auprc", "mean"),
            std_test_auprc=("test_auprc", "std"),
            mean_valid_brier=("valid_brier", "mean"),
            mean_params=("parameter_count_trainable", "mean"),
        )
        .reset_index()
        .sort_values(["mean_valid_auprc"], ascending=False)
    )
    summary.to_csv(summary_path, index=False)

    manifest_path.write_text(
        "\n".join(
            [
                f"Wave: 2 — architecture / stage={stage}",
                f"Date folder: {day}{suffix}",
                "Frozen: clean | empirical | candidate_seed=20260722 | no HCR",
                "Models: " + ", ".join(sorted(frame["model"].astype(str).unique())),
                "Seeds: " + ", ".join(map(str, sorted(frame["training_seed"].unique()))),
                f"n_runs: {len(frame)}",
                "",
                f"CSV: {all_path.name}",
                f"Summary: {summary_path.name}",
                "Per-model: by_model/",
                "",
                "Wave1 (baselines/ablations): outputs/wave1_baselines_ablations/",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {all_path}")
    print(f"Wrote {summary_path}")
    return out_dir


def organize_wave1_snapshot(day: str | None = None) -> Path:
    day = day or date.today().isoformat()
    out_dir = PROJECT_ROOT / "outputs" / "wave1_baselines_ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "taskA_BAZA_analysis/taskA_BAZA_results.csv": f"{day}_wave1_BAZA_results.csv",
        "taskA_BAZA_analysis/taskA_BAZA_frozen_depth.csv": f"{day}_wave1_BAZA_frozen_depth.csv",
        "taskA_BAZA_analysis/taskA_BAZA_scenario_drop.csv": f"{day}_wave1_BAZA_scenario_drop.csv",
        "taskA_frozen_feature_ablation.csv": f"{day}_wave1_E1_frozen_feature_ablation.csv",
        "taskA_frozen_feature_ablation_summary.csv": f"{day}_wave1_E1_frozen_feature_ablation_summary.csv",
        "taskA_node_type_permutation.csv": f"{day}_wave1_E2_node_type_permutation.csv",
        "taskA_node_type_permutation_summary.csv": f"{day}_wave1_E2_node_type_permutation_summary.csv",
        "taskA_per_edge_type.csv": f"{day}_wave1_E3_per_edge_type.csv",
        "taskA_per_edge_type_summary.csv": f"{day}_wave1_E3_per_edge_type_summary.csv",
        "taskA_per_edge_type_summary_exploratory.csv": f"{day}_wave1_E3_per_edge_type_summary_exploratory.csv",
        "taskA_per_edge_type_seed_level.csv": f"{day}_wave1_E3_per_edge_type_seed_level.csv",
        "taskA_relation_ablation.csv": f"{day}_wave1_E4_relation_ablation.csv",
        "taskA_relation_ablation_summary.csv": f"{day}_wave1_E4_relation_ablation_summary.csv",
    }
    copied = []
    for src_rel, dst_name in mapping.items():
        src = PROJECT_ROOT / "outputs" / src_rel
        if src.exists():
            (out_dir / dst_name).write_bytes(src.read_bytes())
            copied.append(dst_name)
    (out_dir / "MANIFEST.txt").write_text(
        "Wave: 1 — baselines and methodological ablations\n"
        f"Snapshot date: {day}\n"
        "NOT architecture-attention comparison (that is wave2).\n"
        + "\n".join(f"  - {n}" for n in copied)
        + "\n",
        encoding="utf-8",
    )
    print(f"Wave1 snapshot → {out_dir} ({len(copied)} files)")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", default=None)
    parser.add_argument(
        "--stage",
        default="ARCH_SCREENING",
        choices=sorted(STAGE_TO_DIR_SUFFIX),
    )
    parser.add_argument("--wave1-snapshot", action="store_true")
    parser.add_argument(
        "--all-stages",
        action="store_true",
        help="Export SCREENING + DEPTH + HEADS + FINAL if present.",
    )
    args = parser.parse_args()
    if args.wave1_snapshot:
        organize_wave1_snapshot(args.day)
    if args.all_stages:
        for stage in STAGE_TO_DIR_SUFFIX:
            write_stage_exports(stage, args.day)
    else:
        write_stage_exports(args.stage, args.day)


if __name__ == "__main__":
    main()
