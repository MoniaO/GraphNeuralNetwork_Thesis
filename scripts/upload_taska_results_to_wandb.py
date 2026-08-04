#!/usr/bin/env python3
"""Upload local Task A aggregate result tables to Weights & Biases.

This does not retrain models. Fixed run IDs make repeated uploads resumable;
rerun after K1 full scenarios and the rare-endpoint audit finish.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import wandb

ROOT = Path(__file__).resolve().parents[1]
WAVE = ROOT / "outputs" / "wave11_taskA"
DIRECT = WAVE / "mlp_vs_kan_full"
ARCH = WAVE / "kan_architecture_audit"
PROJECT = "politechnika-gnn-thesis"
ENTITY = "politechnika-gnn-thesis"
GROUP = "TaskA_KAN_ARCHITECTURE_AUDIT"


def numeric_summary(payload: dict[str, Any], prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in payload.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(numeric_summary(value, name))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            out[name] = float(value)
    return out


def available_tables(paths: list[Path]) -> dict[str, pd.DataFrame]:
    tables = {}
    for path in paths:
        if path.exists() and path.stat().st_size:
            tables[path.stem] = pd.read_csv(path)
    return tables


def upload_run(
    *,
    run_id: str,
    name: str,
    phase: str,
    tables: dict[str, pd.DataFrame],
    files: list[Path],
    summary_json: Path | None = None,
    status: str,
) -> str:
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        id=run_id,
        resume="allow",
        name=name,
        group=GROUP,
        job_type="aggregate_results",
        tags=["TaskA", "link_prediction", "KAN", phase],
        config={
            "task": "Task A structural link prediction",
            "phase": phase,
            "status": status,
            "selection_metric": "validation AUPRC",
            "test_policy": "report-only",
        },
        reinit=True,
    )
    assert run is not None
    for table_name, frame in tables.items():
        run.log({f"tables/{table_name}": wandb.Table(dataframe=frame)})

    if summary_json is not None and summary_json.exists():
        payload = json.loads(summary_json.read_text())
        for key, value in numeric_summary(payload).items():
            run.summary[key] = value

    artifact = wandb.Artifact(
        name=f"{run_id}-tables",
        type="taska-results",
        metadata={"phase": phase, "status": status},
    )
    added = 0
    for path in files:
        if path.exists():
            artifact.add_file(str(path), name=path.name)
            added += 1
    if added:
        run.log_artifact(artifact)

    run.summary["status"] = status
    run.summary["n_tables"] = len(tables)
    url = run.url
    run.finish()
    return str(url)


def scenario_progress_table() -> pd.DataFrame:
    rows = []
    base = ARCH / "scenarios" / "runs" / "TA_KAN_SHALLOW"
    for metrics_path in base.rglob("metrics.json") if base.exists() else []:
        rows.append(json.loads(metrics_path.read_text()))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("all", "direct", "screen", "multiseed", "k1-full", "rare"),
        default="all",
    )
    args = parser.parse_args()
    urls = {}

    if args.phase in {"all", "direct"}:
        files = [
            DIRECT / "global_metrics.csv",
            DIRECT / "scenario_summary.csv",
            DIRECT / "endpoint_path_metrics.csv",
            DIRECT / "endpoint_path_summary.csv",
            DIRECT / "FINAL_DECISION.json",
            DIRECT / "DIRECT_KAN_DECISION.json",
        ]
        urls["direct"] = upload_run(
            run_id="taska-direct-mlp-vs-kan-full",
            name="TaskA Direct MLP vs KAN — 36 runs",
            phase="direct_36",
            tables=available_tables([p for p in files if p.suffix == ".csv"]),
            files=files,
            summary_json=DIRECT / "FINAL_DECISION.json",
            status="complete",
        )

    if args.phase in {"all", "screen"}:
        files = [
            ARCH / "architecture_summary.csv",
            ARCH / "SCREEN_PROMOTION.json",
            ARCH / "FINAL_KAN_ARCHITECTURE_DECISION.json",
        ]
        urls["screen"] = upload_run(
            run_id="taska-kan-architecture-screen",
            name="TaskA KAN-ARCH — K1–K5 screen",
            phase="architecture_screen",
            tables=available_tables([ARCH / "architecture_summary.csv"]),
            files=files,
            summary_json=ARCH / "SCREEN_PROMOTION.json",
            status="complete",
        )

    if args.phase in {"all", "multiseed"}:
        files = [
            ARCH / "multiseed" / "multiseed_summary.csv",
            ARCH / "MULTISEED_STATUS.json",
            ARCH / "SCREEN_PROMOTION.json",
        ]
        urls["multiseed"] = upload_run(
            run_id="taska-kan-architecture-multiseed",
            name="TaskA KAN-ARCH — K1/K2 multiseed",
            phase="architecture_multiseed",
            tables=available_tables(
                [ARCH / "multiseed" / "multiseed_summary.csv"]
            ),
            files=files,
            summary_json=ARCH / "MULTISEED_STATUS.json",
            status="complete",
        )

    if args.phase in {"all", "k1-full"}:
        progress = scenario_progress_table()
        status = "complete" if len(progress) == 18 else f"in_progress_{len(progress)}_of_18"
        tables = {"k1_full_scenario_progress": progress} if len(progress) else {}
        files = [
            ARCH / "scenarios" / "k1_full_scenarios.csv",
            ARCH / "FULL_SCENARIO_STATUS.json",
        ]
        urls["k1-full"] = upload_run(
            run_id="taska-kan-shallow-full-scenarios",
            name="TaskA K1 Shallow KAN — full scenarios",
            phase="k1_full_scenarios",
            tables=tables,
            files=files,
            summary_json=ARCH / "FULL_SCENARIO_STATUS.json",
            status=status,
        )

    if args.phase in {"all", "rare"}:
        rare = ARCH / "endpoint_path"
        files = [
            rare / "MLP_VS_KAN_RARE_ENDPOINT_TABLE.csv",
            rare / "MLP_VS_KAN_RARITY_SUMMARY.csv",
            rare / "winner_endpoint_path_metrics_raw.csv",
            rare / "RARE_ENDPOINT_AUDIT_MANIFEST.json",
        ]
        if any(path.exists() for path in files):
            urls["rare"] = upload_run(
                run_id="taska-mlp-vs-kan-rare-endpoint-paths",
                name="TaskA MLP vs KAN — rare endpoint paths",
                phase="rare_endpoint_path_audit",
                tables=available_tables([p for p in files if p.suffix == ".csv"]),
                files=files,
                summary_json=rare / "RARE_ENDPOINT_AUDIT_MANIFEST.json",
                status="complete",
            )

    print(json.dumps(urls, indent=2))


if __name__ == "__main__":
    main()
