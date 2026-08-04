#!/usr/bin/env python3
"""Orchestrate WAVE 5D — Patient-Conditioned Path-Supported Link Prediction."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
os.environ.setdefault(
    "GSN_PROJECT_ROOT",
    str(Path.home() / "Desktop" / "GSN Graphs dysertation 2026"),
)
os.environ["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT}:{os.environ.get('PYTHONPATH', '')}"


def run(cmd: list[str]) -> None:
    print(">>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main() -> None:
    cfg = "configs/link_prediction/wave5d_clean.yaml"
    out = ROOT / "outputs/wave5d_link"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "wave5d_console.log"
    print("=" * 60)
    print("WAVE 5D — PATIENT-CONDITIONED PATH-SUPPORTED LINK PREDICTION")
    print("=" * 60)
    print("Started:", datetime.now())

    steps = [
        [PY, "scripts/build_wave5d_candidates.py", "--config", cfg],
        [PY, "scripts/run_wave5d_ablations.py", "--config", cfg],
        [PY, "scripts/summarize_wave5d_link.py", "--config", cfg],
        [PY, "scripts/run_wave5d_scenarios.py"],
    ]
    with log.open("w") as fh:
        for cmd in steps:
            print(">>>", " ".join(cmd), flush=True)
            fh.write(">>> " + " ".join(cmd) + "\n")
            fh.flush()
            proc = subprocess.run(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            sys.stdout.write(proc.stdout)
            fh.write(proc.stdout)
            fh.flush()
            if proc.returncode != 0:
                raise SystemExit(proc.returncode)

    print("Finished:", datetime.now())


if __name__ == "__main__":
    main()
