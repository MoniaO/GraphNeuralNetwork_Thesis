#!/usr/bin/env python3
"""Compare final MLP and selected KAN on rare endpoint-path edge subsets."""

from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
DEFAULT_GSN = Path.home() / "Desktop" / "GSN Graphs dysertation 2026"
WAVE = ROOT / "outputs" / "wave11_taskA"
DIRECT = WAVE / "mlp_vs_kan_full"
ARCH = WAVE / "kan_architecture_audit"
OUT = ARCH / "endpoint_path"
SCENARIOS = (
    "clean",
    "hidden_confounder",
    "selection_bias",
    "no_overlap",
    "noisy_documentation",
    "multihospital",
)
SEEDS = (20260722, 20260723, 20260724)
MODELS = ("TA_MLP", "TA_KAN_SHALLOW")


def rarity_group(prevalence: float) -> str:
    if prevalence < 0.01:
        return "ultra_sparse_<1pct"
    if prevalence < 0.05:
        return "sparse_1_5pct"
    if prevalence < 0.10:
        return "moderate_5_10pct"
    return "common_>=10pct"


def patient_endpoint_prevalence() -> dict[tuple[str, str], float]:
    """Train-patient endpoint prevalence per scenario (audit metadata only)."""
    gsn = Path(os.environ.get("GSN_PROJECT_ROOT", DEFAULT_GSN))
    data_dir = gsn / "2 v3. Data" / "dataset_v3"
    split_path = gsn / "2 v3. Data" / "splits" / "patient_splits_v3.csv"
    splits = pd.read_csv(split_path)
    train_ids = set(
        splits.loc[splits["split"].astype(str).eq("train"), "patient_id"].astype(int)
    )
    endpoints = (
        "AKI",
        "DILI",
        "Depression",
        "Falls",
        "Delirium",
        "GI_bleeding",
        "Hyponatremia",
        "Hyperkalemia",
        "QT_arrhythmia",
        "Hospitalization",
    )
    out = {}
    for scenario in SCENARIOS:
        path = data_dir / f"synthetic_pharmacotherapy_v3_samples_{scenario}.csv"
        frame = pd.read_csv(path, usecols=["patient_id", *endpoints])
        frame = frame[frame["patient_id"].astype(int).isin(train_ids)]
        for endpoint in endpoints:
            out[(scenario, endpoint)] = float(
                pd.to_numeric(frame[endpoint], errors="coerce").mean()
            )
    return out


def ensure_k1_endpoint_metrics() -> None:
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", f"{ROOT / 'src'}:{ROOT}")
    for scenario in SCENARIOS:
        for seed in SEEDS:
            run_dir = (
                ARCH
                / "scenarios"
                / "runs"
                / "TA_KAN_SHALLOW"
                / scenario
                / f"seed_{seed}"
            )
            checkpoint = run_dir / "best_model.pt"
            output = run_dir / "endpoint_path_metrics.csv"
            if output.exists():
                continue
            if not checkpoint.exists():
                raise SystemExit(f"Missing K1 checkpoint: {checkpoint}")
            cmd = [
                PY,
                str(ROOT / "scripts" / "eval_taska_run_endpoint_paths.py"),
                "--model=TA_KAN_SHALLOW",
                f"--scenario={scenario}",
                f"--seed={seed}",
            ]
            print("EVAL:", " ".join(cmd), flush=True)
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
            if proc.returncode != 0:
                raise SystemExit(
                    f"Endpoint audit failed: {scenario} seed={seed}, rc={proc.returncode}"
                )


def metric_path(model: str, scenario: str, seed: int) -> Path:
    if model == "TA_MLP":
        return (
            DIRECT
            / "runs"
            / model
            / scenario
            / f"seed_{seed}"
            / "endpoint_path_metrics.csv"
        )
    return (
        ARCH
        / "scenarios"
        / "runs"
        / model
        / scenario
        / f"seed_{seed}"
        / "endpoint_path_metrics.csv"
    )


def collect() -> pd.DataFrame:
    frames = []
    for model in MODELS:
        for scenario in SCENARIOS:
            for seed in SEEDS:
                path = metric_path(model, scenario, seed)
                if not path.exists():
                    raise SystemExit(f"Missing endpoint metrics: {path}")
                frame = pd.read_csv(path)
                frame = frame[frame["hospital"].astype(str).eq("pooled")].copy()
                frame["model"] = model
                frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def mean_sd(values: pd.Series) -> tuple[float, float]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) == 0:
        return float("nan"), float("nan")
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def summarize_valid(raw: pd.DataFrame) -> pd.DataFrame:
    valid = raw[raw["split"].eq("valid")].copy()
    patient_prev = patient_endpoint_prevalence()
    rows = []
    keys = ["scenario", "endpoint"]
    for (scenario, endpoint), group in valid.groupby(keys, sort=True):
        mlp = group[group["model"].eq("TA_MLP")].set_index("seed")
        kan = group[group["model"].eq("TA_KAN_SHALLOW")].set_index("seed")
        common = sorted(set(mlp.index) & set(kan.index))
        if not common:
            continue
        mlp = mlp.loc[common]
        kan = kan.loc[common]
        edge_prevalence = float(mlp["edge_prevalence"].mean())
        endpoint_prevalence = patient_prev[(scenario, endpoint)]
        mlp_ap_m, mlp_ap_sd = mean_sd(mlp["auprc"])
        kan_ap_m, kan_ap_sd = mean_sd(kan["auprc"])
        mlp_n_m, mlp_n_sd = mean_sd(mlp["normalized_auprc"])
        kan_n_m, kan_n_sd = mean_sd(kan["normalized_auprc"])
        delta_ap = kan["auprc"].to_numpy() - mlp["auprc"].to_numpy()
        delta_n = (
            kan["normalized_auprc"].to_numpy()
            - mlp["normalized_auprc"].to_numpy()
        )
        rows.append(
            {
                "scenario": scenario,
                "endpoint": endpoint,
                "rarity_group": rarity_group(endpoint_prevalence),
                "n_seeds": len(common),
                "n_candidate_edges_mean": float(mlp["n_candidate_edges"].mean()),
                "n_positive_edges_mean": float(mlp["n_positive_edges"].mean()),
                "patient_endpoint_prevalence_train": endpoint_prevalence,
                "edge_prevalence_mean": edge_prevalence,
                "mlp_valid_auprc_mean": mlp_ap_m,
                "mlp_valid_auprc_sd": mlp_ap_sd,
                "kan_valid_auprc_mean": kan_ap_m,
                "kan_valid_auprc_sd": kan_ap_sd,
                "delta_kan_minus_mlp_auprc_mean": float(np.mean(delta_ap)),
                "delta_kan_minus_mlp_auprc_sd": float(np.std(delta_ap, ddof=1)),
                "kan_seed_wins_auprc": int(np.sum(delta_ap > 0)),
                "mlp_valid_nauprc_mean": mlp_n_m,
                "mlp_valid_nauprc_sd": mlp_n_sd,
                "kan_valid_nauprc_mean": kan_n_m,
                "kan_valid_nauprc_sd": kan_n_sd,
                "delta_kan_minus_mlp_nauprc_mean": float(np.mean(delta_n)),
                "delta_kan_minus_mlp_nauprc_sd": float(np.std(delta_n, ddof=1)),
                "kan_seed_wins_nauprc": int(np.sum(delta_n > 0)),
            }
        )
    return pd.DataFrame(rows)


def rarity_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rarity, group in table.groupby("rarity_group", sort=True):
        rows.append(
            {
                "rarity_group": rarity,
                "n_scenario_endpoint_cells": len(group),
                "n_positive_edges_mean": group["n_positive_edges_mean"].mean(),
                "patient_endpoint_prevalence_train_mean": group[
                    "patient_endpoint_prevalence_train"
                ].mean(),
                "edge_prevalence_mean": group["edge_prevalence_mean"].mean(),
                "mlp_macro_valid_auprc": group["mlp_valid_auprc_mean"].mean(),
                "kan_macro_valid_auprc": group["kan_valid_auprc_mean"].mean(),
                "delta_kan_minus_mlp_macro_auprc": group[
                    "delta_kan_minus_mlp_auprc_mean"
                ].mean(),
                "mlp_macro_valid_nauprc": group["mlp_valid_nauprc_mean"].mean(),
                "kan_macro_valid_nauprc": group["kan_valid_nauprc_mean"].mean(),
                "delta_kan_minus_mlp_macro_nauprc": group[
                    "delta_kan_minus_mlp_nauprc_mean"
                ].mean(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_k1_endpoint_metrics()
    OUT.mkdir(parents=True, exist_ok=True)
    raw = collect()
    raw.to_csv(OUT / "winner_endpoint_path_metrics_raw.csv", index=False)
    comparison = summarize_valid(raw)
    comparison.to_csv(OUT / "MLP_VS_KAN_RARE_ENDPOINT_TABLE.csv", index=False)
    rare = rarity_summary(comparison)
    rare.to_csv(OUT / "MLP_VS_KAN_RARITY_SUMMARY.csv", index=False)
    decision = {
        "task": "Task A link prediction",
        "models": {
            "mlp_winner": "TA_MLP / W7D_T0_A1",
            "kan_winner": "TA_KAN_SHALLOW",
        },
        "rarity_definition": (
            "endpoint prevalence among training patients in each scenario; "
            "candidate-edge prevalence is reported separately"
        ),
        "selection_split": "valid",
        "test_policy": "report_only in raw table",
        "n_scenario_endpoint_cells": int(len(comparison)),
        "outputs": [
            "MLP_VS_KAN_RARE_ENDPOINT_TABLE.csv",
            "MLP_VS_KAN_RARITY_SUMMARY.csv",
            "winner_endpoint_path_metrics_raw.csv",
        ],
    }
    (OUT / "RARE_ENDPOINT_AUDIT_MANIFEST.json").write_text(
        json.dumps(decision, indent=2)
    )
    print(f"wrote winner comparison to {OUT}", flush=True)


if __name__ == "__main__":
    main()
