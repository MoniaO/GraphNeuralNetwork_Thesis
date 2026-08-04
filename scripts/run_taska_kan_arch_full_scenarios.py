#!/usr/bin/env python3
"""Etap 3: full 6-scenario validation of the selected K1 shallow KAN."""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_taska_kan_architecture_audit import (  # noqa: E402
    DIRECT_OUT,
    MULTISEED_SEEDS,
    OUT,
    run_one,
)

ARCH_KEY = "k1_shallow"
ARCHITECTURE = "TA_KAN_SHALLOW"
SCENARIOS = (
    "clean",
    "hidden_confounder",
    "selection_bias",
    "no_overlap",
    "noisy_documentation",
    "multihospital",
)


def reuse_multiseed(scenario: str, seed: int) -> dict | None:
    source = (
        OUT
        / "multiseed"
        / "runs"
        / ARCHITECTURE
        / scenario
        / f"seed_{seed}"
    )
    target = (
        OUT
        / "scenarios"
        / "runs"
        / ARCHITECTURE
        / scenario
        / f"seed_{seed}"
    )
    if not (source / "metrics.json").exists() or not (source / "best_model.pt").exists():
        return None
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "best_model.pt", target / "best_model.pt")
    row = json.loads((source / "metrics.json").read_text())
    row["reused_from"] = str(source.relative_to(ROOT))
    (target / "metrics.json").write_text(json.dumps(row, indent=2, default=float))
    print(f"REUSE multiseed {scenario} seed={seed}", flush=True)
    return row


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()

    rows = []
    for scenario in SCENARIOS:
        for seed in MULTISEED_SEEDS:
            row = reuse_multiseed(scenario, seed)
            if row is None:
                row = run_one(
                    ARCH_KEY,
                    scenario,
                    seed,
                    args.epochs,
                    stage="scenarios",
                )
            rows.append(row)

    out_csv = OUT / "scenarios" / "k1_full_scenarios.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "architecture",
        "scenario",
        "seed",
        "valid_auprc",
        "test_auprc",
        "delta_vs_mlp_same_seed",
        "parameter_count",
        "runtime_s",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            mlp_path = (
                DIRECT_OUT
                / "runs"
                / "TA_MLP"
                / row["scenario"]
                / f"seed_{row['seed']}"
                / "metrics.json"
            )
            mlp = json.loads(mlp_path.read_text())
            writer.writerow(
                {
                    **{key: row.get(key) for key in fields},
                    "delta_vs_mlp_same_seed": (
                        float(row["valid_auprc"]) - float(mlp["valid_auprc"])
                    ),
                }
            )

    status = {
        "wave": "TASKA_KAN_ARCH_FULL_SCENARIOS",
        "architecture": ARCHITECTURE,
        "scenarios": list(SCENARIOS),
        "seeds": list(MULTISEED_SEEDS),
        "n_runs": len(rows),
        "selection_metric": "valid_auprc",
        "test_policy": "report_only",
    }
    (OUT / "FULL_SCENARIO_STATUS.json").write_text(json.dumps(status, indent=2))
    print(f"wrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
