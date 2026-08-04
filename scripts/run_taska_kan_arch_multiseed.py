#!/usr/bin/env python3
"""Etap 2: multiseed for promoted KAN-ARCH variants (clean + multihospital)."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_taska_kan_architecture_audit import (  # noqa: E402
    DIRECT_OUT,
    MULTISEED_SEEDS,
    OUT,
    SCREEN_SCENARIOS,
    SCREEN_SEED,
    run_one,
)

SUMMARY = OUT / "architecture_summary.csv"
KEY = {
    "TA_KAN_SHALLOW": "k1_shallow",
    "TA_MLP_TO_KAN": "k2_mlp_to_kan",
    "TA_KAN_TO_LINEAR": "k3_kan_to_linear",
    "TA_LINEAR_PLUS_KAN": "k4_linear_plus_kan",
    "TA_GROUPED_KAN": "k5_grouped_kan",
}


def promote() -> list[str]:
    by = defaultdict(list)
    with SUMMARY.open() as f:
        for row in csv.DictReader(f):
            by[row["architecture"]].append(float(row["delta_vs_mlp"]))
    promoted, details = [], {}
    for arch, deltas in by.items():
        mean_d = sum(deltas) / len(deltas)
        worst = min(deltas)
        ok = mean_d >= -0.005 and worst >= -0.05
        details[arch] = {"deltas": deltas, "mean_delta": mean_d, "worst": worst, "promote": ok}
        if ok:
            promoted.append(KEY[arch])
    decision = {
        "wave": "TASKA_KAN_ARCH_SCREEN_PROMOTION",
        "rule": "mean_delta>=-0.005 AND worst_delta>=-0.05",
        "promoted_arch_keys": promoted,
        "details": details,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "SCREEN_PROMOTION.json").write_text(json.dumps(decision, indent=2))
    print(json.dumps(decision, indent=2), flush=True)
    return promoted


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--promote-only", action="store_true")
    args = ap.parse_args()
    promoted = promote()
    if args.promote_only:
        return
    if not promoted:
        raise SystemExit("no promoted architectures")

    (OUT / "multiseed").mkdir(exist_ok=True)
    rows = []
    for scenario in SCREEN_SCENARIOS:
        for seed in MULTISEED_SEEDS:
            for arch in promoted:
                rows.append(
                    run_one(arch, scenario, seed, args.epochs, stage="multiseed")
                )

    out_csv = OUT / "multiseed" / "multiseed_summary.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "architecture",
                "scenario",
                "seed",
                "valid_auprc",
                "test_auprc",
                "delta_vs_mlp_same_seed",
                "parameter_count",
                "runtime_s",
            ],
        )
        w.writeheader()
        for r in rows:
            mlp_p = (
                DIRECT_OUT
                / "runs"
                / "TA_MLP"
                / r["scenario"]
                / f"seed_{r['seed']}"
                / "metrics.json"
            )
            delta = float("nan")
            if mlp_p.exists():
                delta = float(r["valid_auprc"]) - float(
                    json.loads(mlp_p.read_text())["valid_auprc"]
                )
            w.writerow(
                {
                    "architecture": r["architecture"],
                    "scenario": r["scenario"],
                    "seed": r["seed"],
                    "valid_auprc": r["valid_auprc"],
                    "test_auprc": r["test_auprc"],
                    "delta_vs_mlp_same_seed": delta,
                    "parameter_count": r.get("parameter_count"),
                    "runtime_s": r.get("runtime_s"),
                }
            )
    (OUT / "MULTISEED_STATUS.json").write_text(
        json.dumps(
            {
                "wave": "TASKA_KAN_ARCH_MULTISEED",
                "promoted": promoted,
                "n_runs": len(rows),
                "scenarios": list(SCREEN_SCENARIOS),
                "seeds": list(MULTISEED_SEEDS),
            },
            indent=2,
        )
    )
    print(f"wrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
