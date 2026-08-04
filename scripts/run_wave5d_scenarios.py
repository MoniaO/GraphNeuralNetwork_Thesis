#!/usr/bin/env python3
"""Wave 5E scenario matrix stub — full runs after clean WAVE5D decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/link_prediction/wave5d_scenarios.yaml",
    )
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = ROOT / "outputs/wave5d_link/summary"
    out.mkdir(parents=True, exist_ok=True)
    note = {
        "status": "deferred_to_wave5e",
        "message": (
            "Clean WAVE5D pilot must finish first. "
            "Scenario in/cross evaluation is Wave 5E."
        ),
        "scenarios": cfg.get("scenarios", []),
        "modes": cfg.get("evaluation_modes", []),
    }
    path = out / "scenario_robustness.csv"
    # Placeholder empty frame marker
    path.write_text("scenario,mode,status\n")
    (out / "SCENARIO_STATUS.json").write_text(json.dumps(note, indent=2))
    print(json.dumps(note, indent=2))


if __name__ == "__main__":
    main()
