#!/usr/bin/env python3
"""Orchestrate WAVE 5C — Patient-Conditioned Mechanistic Path Activation."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

PYTHON = str(ROOT / ".venv" / "bin" / "python")
if not Path(PYTHON).exists():
    PYTHON = sys.executable

ENV = os.environ.copy()
ENV["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT}:{ENV.get('PYTHONPATH', '')}"
ENV["PYTHONUNBUFFERED"] = "1"
ENV.setdefault(
    "GSN_PROJECT_ROOT",
    str(Path.home() / "Desktop" / "GSN Graphs dysertation 2026"),
)

LOG = ROOT / "outputs/wave5c" / "wave5c_runner.log"


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = msg if msg.endswith("\n") else msg + "\n"
    with LOG.open("a") as f:
        f.write(line)
    print(line, end="", flush=True)


def run(cmd: list[str]) -> None:
    log(">>> " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(ROOT), env=ENV)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> None:
    log("=" * 60)
    log("WAVE 5C — PATIENT-CONDITIONED MECHANISTIC PATH ACTIVATION")
    log(f"Started: {datetime.now()}")
    log("=" * 60)
    run([PYTHON, "scripts/build_wave5c_context_registry.py"])
    run([PYTHON, "scripts/build_wave5c_patient_pairs.py"])
    run([PYTHON, "scripts/tune_wave5c_energy.py"])
    run([PYTHON, "scripts/run_wave5c.py", "--split", "test"])
    run([PYTHON, "scripts/summarize_wave5c.py"])
    log(f"Finished: {datetime.now()}")


if __name__ == "__main__":
    main()
