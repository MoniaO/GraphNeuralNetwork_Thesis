#!/usr/bin/env python3
"""Run Wave 5D feature extraction + training for all variants × seeds."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

CONTROL_EXTRACT = {
    "none": ["none"],
    "controls": [
        "patient_shuffle",
        "context_shuffle",
        "matched_random_context",
        "random_path_weights",
        "no_leave_one_out",
    ],
}


def run(cmd: list[str]) -> None:
    print(">>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/link_prediction/wave5d_clean.yaml",
    )
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    seeds = args.seeds or list(cfg["seeds"])
    variants = args.variants or list(cfg["variants"])

    if not args.skip_extract:
        for seed in seeds:
            run(
                [
                    PY,
                    "scripts/extract_patient_path_support.py",
                    "--config",
                    str(args.config),
                    "--seed",
                    str(seed),
                    "--control",
                    "none",
                ]
            )
            for control in CONTROL_EXTRACT["controls"]:
                # Only extract controls needed by selected variants
                needed = {
                    "L5_patient_shuffle": "patient_shuffle",
                    "L6_context_shuffle": "context_shuffle",
                    "L7_matched_random_context": "matched_random_context",
                    "L8_random_path_weights": "random_path_weights",
                    "L10_no_leave_one_out": "no_leave_one_out",
                }
                if control not in needed.values():
                    continue
                if not any(needed.get(v) == control for v in variants):
                    continue
                run(
                    [
                        PY,
                        "scripts/extract_patient_path_support.py",
                        "--config",
                        str(args.config),
                        "--seed",
                        str(seed),
                        "--control",
                        control,
                    ]
                )

    for seed in seeds:
        for variant in variants:
            run(
                [
                    PY,
                    "scripts/train_wave5d_link_predictor.py",
                    "--config",
                    str(args.config),
                    "--seed",
                    str(seed),
                    "--variant",
                    variant,
                ]
            )


if __name__ == "__main__":
    main()
