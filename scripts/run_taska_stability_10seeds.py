#!/usr/bin/env python3
"""Ten-seed stability study: final Task A MLP versus K1 shallow KAN.

The protocol pairs both models on the same scenario, candidate seed, training
seed, and data fingerprint. Existing Wave 11 runs are reused when available;
missing runs are resumable and written to outputs/wave11_taskA/stability_10seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import yaml
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_taska_kan_architecture_audit import (  # noqa: E402
    _env as kan_env,
    _metrics_from_ckpt as kan_metrics_from_ckpt,
    build_cmd as build_kan_cmd,
)
from run_taska_mlp_vs_kan_scenarios import (  # noqa: E402
    _env as mlp_env,
    _metrics_from_ckpt as mlp_metrics_from_ckpt,
    build_train_cmd as build_mlp_cmd,
)

OUT = ROOT / "outputs" / "wave11_taskA" / "stability_10seeds"
DIRECT_OUT = ROOT / "outputs" / "wave11_taskA" / "mlp_vs_kan_full"
KAN_AUDIT_OUT = ROOT / "outputs" / "wave11_taskA" / "kan_architecture_audit"
MLP_CONFIG = ROOT / "configs" / "taskA" / "mlp_full_retrain.yaml"
KAN_CONFIG = ROOT / "configs" / "taskA" / "kan_architectures" / "k1_shallow.yaml"

MODELS = ("TA_MLP", "TA_KAN_SHALLOW")
SCENARIOS = (
    "clean",
    "hidden_confounder",
    "selection_bias",
    "no_overlap",
    "noisy_documentation",
    "multihospital",
)
DEFAULT_SEEDS = tuple(range(20260722, 20260732))


def _normalize(row: dict, model: str, scenario: str, seed: int) -> dict:
    return {
        "model": model,
        "scenario": scenario,
        "seed": seed,
        "valid_auprc": float(row["valid_auprc"]),
        "test_auprc": float(row["test_auprc"]),
        "valid_auroc": float(row.get("valid_auroc", float("nan"))),
        "test_auroc": float(row.get("test_auroc", float("nan"))),
        "valid_brier": float(row.get("valid_brier", float("nan"))),
        "test_brier": float(row.get("test_brier", float("nan"))),
        "best_epoch": row.get("best_epoch"),
        "parameter_count": row.get("parameter_count"),
        "candidate_fingerprint": row.get("candidate_fingerprint"),
        "runtime_s": float(row.get("runtime_s", 0.0)),
        "reused_from": row.get("reused_from", ""),
    }


def _existing_sources(model: str, scenario: str, seed: int) -> list[Path]:
    local = OUT / "runs" / model / scenario / f"seed_{seed}"
    if model == "TA_MLP":
        historical = DIRECT_OUT / "runs" / model / scenario / f"seed_{seed}"
        return [local, historical]
    scenario_run = (
        KAN_AUDIT_OUT
        / "scenarios"
        / "runs"
        / model
        / scenario
        / f"seed_{seed}"
    )
    multiseed_run = (
        KAN_AUDIT_OUT
        / "multiseed"
        / "runs"
        / model
        / scenario
        / f"seed_{seed}"
    )
    return [local, scenario_run, multiseed_run]


def load_existing(model: str, scenario: str, seed: int) -> dict | None:
    for run_dir in _existing_sources(model, scenario, seed):
        metrics_path = run_dir / "metrics.json"
        checkpoint = run_dir / "best_model.pt"
        if metrics_path.exists() and checkpoint.exists():
            row = json.loads(metrics_path.read_text())
            row["reused_from"] = str(run_dir.relative_to(ROOT))
            print(f"REUSE {model} {scenario} seed={seed}", flush=True)
            return _normalize(row, model, scenario, seed)
    return None


def _run_logged(cmd: list[str], env: dict, run_dir: Path) -> float:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    started = time.time()
    with log_path.open("w") as log:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.time() - started
    if proc.returncode != 0:
        excerpt = "\n".join(log_path.read_text(errors="replace").splitlines()[-40:])
        raise RuntimeError(
            f"training failed with rc={proc.returncode}: {run_dir}\n{excerpt}"
        )
    return elapsed


def train_missing(
    model: str,
    scenario: str,
    seed: int,
    epochs: int,
) -> dict:
    run_dir = OUT / "runs" / model / scenario / f"seed_{seed}"
    print(f"RUN {model} {scenario} seed={seed}", flush=True)

    if model == "TA_MLP":
        cfg = yaml.safe_load(MLP_CONFIG.read_text())
        cmd = build_mlp_cmd(
            model_id=model,
            cfg=cfg,
            scenario=scenario,
            seed=seed,
            epochs=epochs,
            run_dir=run_dir,
        )
        elapsed = _run_logged(cmd, mlp_env(), run_dir)
        metrics = mlp_metrics_from_ckpt(run_dir / "best_model.pt")
    else:
        cmd = build_kan_cmd(KAN_CONFIG, scenario, seed, epochs, run_dir)
        elapsed = _run_logged(cmd, kan_env(), run_dir)
        metrics = kan_metrics_from_ckpt(run_dir / "best_model.pt")

    row = _normalize(
        {**metrics, "runtime_s": elapsed},
        model,
        scenario,
        seed,
    )
    (run_dir / "metrics.json").write_text(json.dumps(row, indent=2, default=float))
    print(
        f"DONE {model} {scenario} seed={seed} "
        f"valid={row['valid_auprc']:.6f} test={row['test_auprc']:.6f} "
        f"runtime={elapsed:.1f}s",
        flush=True,
    )
    return row


def collect_rows(scenarios: tuple[str, ...], seeds: tuple[int, ...]) -> list[dict]:
    rows: list[dict] = []
    for scenario in scenarios:
        for seed in seeds:
            for model in MODELS:
                row = load_existing(model, scenario, seed)
                if row is not None:
                    rows.append(row)
    return rows


def _mean_sd(values: pd.Series) -> tuple[float, float]:
    return float(values.mean()), float(values.std(ddof=1))


def write_summaries(
    rows: list[dict],
    scenarios: tuple[str, ...],
    seeds: tuple[int, ...],
) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).drop_duplicates(
        subset=["model", "scenario", "seed"], keep="last"
    )
    frame = frame.sort_values(["scenario", "seed", "model"])
    frame.to_csv(OUT / "run_metrics.csv", index=False)

    summary_rows = []
    for (scenario, model), group in frame.groupby(["scenario", "model"]):
        valid_mean, valid_sd = _mean_sd(group["valid_auprc"])
        test_mean, test_sd = _mean_sd(group["test_auprc"])
        summary_rows.append(
            {
                "scenario": scenario,
                "model": model,
                "n_seeds": len(group),
                "valid_auprc_mean": valid_mean,
                "valid_auprc_sd": valid_sd,
                "valid_auprc_cv": valid_sd / valid_mean if valid_mean else float("nan"),
                "valid_auprc_min": float(group["valid_auprc"].min()),
                "valid_auprc_max": float(group["valid_auprc"].max()),
                "test_auprc_mean": test_mean,
                "test_auprc_sd": test_sd,
                "test_auprc_cv": test_sd / test_mean if test_mean else float("nan"),
                "test_auprc_min": float(group["test_auprc"].min()),
                "test_auprc_max": float(group["test_auprc"].max()),
            }
        )
    pd.DataFrame(summary_rows).to_csv(
        OUT / "scenario_stability_summary.csv", index=False
    )

    paired = frame.pivot_table(
        index=["scenario", "seed"],
        columns="model",
        values=["valid_auprc", "test_auprc"],
        aggfunc="first",
    )
    paired.columns = [f"{metric}_{model}" for metric, model in paired.columns]
    paired = paired.reset_index()
    required = [
        "valid_auprc_TA_MLP",
        "valid_auprc_TA_KAN_SHALLOW",
        "test_auprc_TA_MLP",
        "test_auprc_TA_KAN_SHALLOW",
    ]
    if all(column in paired for column in required):
        paired = paired.dropna(subset=required)
        paired["delta_valid_k1_minus_mlp"] = (
            paired["valid_auprc_TA_KAN_SHALLOW"]
            - paired["valid_auprc_TA_MLP"]
        )
        paired["delta_test_k1_minus_mlp"] = (
            paired["test_auprc_TA_KAN_SHALLOW"]
            - paired["test_auprc_TA_MLP"]
        )
    paired.to_csv(OUT / "paired_runs.csv", index=False)

    expected_pairs = len(scenarios) * len(seeds)
    status = {
        "protocol": "Task A final MLP vs K1 shallow stability",
        "scenarios": list(scenarios),
        "seeds": list(seeds),
        "models": list(MODELS),
        "epochs": 200,
        "completed_model_runs": int(len(frame)),
        "expected_model_runs": int(2 * expected_pairs),
        "completed_pairs": int(len(paired)),
        "expected_pairs": int(expected_pairs),
        "complete": bool(len(frame) == 2 * expected_pairs and len(paired) == expected_pairs),
    }

    if len(paired):
        by_seed = (
            paired.groupby("seed", as_index=False)
            .agg(
                delta_valid_mean=("delta_valid_k1_minus_mlp", "mean"),
                delta_valid_sd_across_scenarios=("delta_valid_k1_minus_mlp", "std"),
                delta_test_mean=("delta_test_k1_minus_mlp", "mean"),
                n_scenarios=("scenario", "count"),
            )
            .sort_values("seed")
        )
        by_seed.to_csv(OUT / "paired_delta_by_seed.csv", index=False)

        deltas = by_seed["delta_valid_mean"]
        n = len(deltas)
        mean_delta = float(deltas.mean())
        sd_delta = float(deltas.std(ddof=1)) if n > 1 else float("nan")
        margin = (
            float(t.ppf(0.975, n - 1)) * sd_delta / math.sqrt(n)
            if n > 1
            else float("nan")
        )
        ci_low, ci_high = mean_delta - margin, mean_delta + margin

        seed_model_means = frame.groupby(["seed", "model"], as_index=False).agg(
            valid_auprc=("valid_auprc", "mean"),
            test_auprc=("test_auprc", "mean"),
        )
        stability = {}
        for model, group in seed_model_means.groupby("model"):
            stability[model] = {
                "n_seeds": int(len(group)),
                "valid_mean_across_seed_means": float(group["valid_auprc"].mean()),
                "valid_sd_across_seed_means": float(group["valid_auprc"].std(ddof=1)),
                "test_mean_across_seed_means": float(group["test_auprc"].mean()),
                "test_sd_across_seed_means": float(group["test_auprc"].std(ddof=1)),
            }

        status["inference"] = {
            "unit": "seed-level mean across six paired scenarios",
            "n_seeds": n,
            "mean_delta_valid_k1_minus_mlp": mean_delta,
            "sd_delta_valid": sd_delta,
            "t95_ci": [ci_low, ci_high],
            "k1_seed_wins": int((deltas > 0).sum()),
            "mlp_seed_wins": int((deltas < 0).sum()),
            "performance_conclusion": (
                "MLP"
                if ci_high < 0
                else ("K1" if ci_low > 0 else "inconclusive")
            ),
            "stability": stability,
        }

    (OUT / "STABILITY_STATUS.json").write_text(
        json.dumps(status, indent=2, default=float)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated integer seeds.",
    )
    parser.add_argument(
        "--scenarios",
        default="all",
        help="Comma-separated scenarios or all.",
    )
    args = parser.parse_args()

    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    scenarios = (
        SCENARIOS
        if args.scenarios == "all"
        else tuple(value.strip() for value in args.scenarios.split(",") if value.strip())
    )
    unknown = sorted(set(scenarios).difference(SCENARIOS))
    if unknown:
        raise SystemExit(f"Unknown scenarios: {unknown}")
    if len(seeds) < 2:
        raise SystemExit("At least two seeds are required for a stability study.")

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": "Task A final MLP vs K1 shallow",
        "models": list(MODELS),
        "scenarios": list(scenarios),
        "seeds": list(seeds),
        "epochs": args.epochs,
        "selection_metric": "validation AUPRC",
        "test_policy": "report only",
        "pairing": "same scenario, candidate seed, training seed, and data fingerprint",
    }
    (OUT / "PROTOCOL.json").write_text(json.dumps(manifest, indent=2))

    rows: list[dict] = []
    expected_pairs = len(scenarios) * len(seeds)
    completed_pairs = 0
    for scenario in scenarios:
        for seed in seeds:
            pair = {}
            for model in MODELS:
                row = load_existing(model, scenario, seed)
                if row is None:
                    row = train_missing(model, scenario, seed, args.epochs)
                pair[model] = row
                rows.append(row)

            mlp_fp = pair["TA_MLP"].get("candidate_fingerprint")
            kan_fp = pair["TA_KAN_SHALLOW"].get("candidate_fingerprint")
            if not mlp_fp or not kan_fp or mlp_fp != kan_fp:
                raise RuntimeError(
                    f"candidate fingerprint mismatch for {scenario} seed={seed}: "
                    f"MLP={mlp_fp}, K1={kan_fp}"
                )
            completed_pairs += 1
            write_summaries(rows, scenarios, seeds)
            print(
                f"PAIR_DONE {completed_pairs}/{expected_pairs} "
                f"{scenario} seed={seed} "
                f"delta={pair['TA_KAN_SHALLOW']['valid_auprc'] - pair['TA_MLP']['valid_auprc']:+.6f}",
                flush=True,
            )

    # Re-scan all reusable/local rows before the final report.
    write_summaries(collect_rows(scenarios, seeds), scenarios, seeds)
    print("STABILITY_10SEEDS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
