#!/usr/bin/env python3
"""Orchestrate WAVE 5B Topology–Energy Decomposition end-to-end."""

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

LOG = ROOT / "outputs/wave5b" / "wave5b_runner.log"


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = msg if msg.endswith("\n") else msg + "\n"
    with LOG.open("a") as f:
        f.write(line)
    print(line, end="", flush=True)


def run(cmd: list[str]) -> None:
    log(">>> " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(ROOT), env=ENV, text=True)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> None:
    log("=" * 60)
    log("WAVE 5B — TOPOLOGY–ENERGY DECOMPOSITION AND CALIBRATION")
    log(f"Started: {datetime.now()}")
    log("=" * 60)

    # Freeze query_split into registry if missing.
    run([PYTHON, "-c", """
from pathlib import Path
import pandas as pd
from wnerw.wave5b.io import load_queries
p = Path('outputs/wave5/queries/wave5_frozen_queries.csv')
q = load_queries(p)
q.to_csv(p, index=False)
print('registry', len(q), q.query_split.value_counts().to_dict())
"""])

    run([PYTHON, "scripts/run_wave5b_topology.py"])
    run([PYTHON, "scripts/tune_wave5b_energy.py"])
    run([PYTHON, "scripts/run_wave5b_energy.py", "--split", "test"])
    run([PYTHON, "scripts/summarize_wave5b.py"])

    log(f"Finished: {datetime.now()}")
    log("Artifacts under outputs/wave5b/")


if __name__ == "__main__":
    main()
