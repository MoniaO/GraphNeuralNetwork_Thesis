#!/usr/bin/env python3
"""05 — Stage C: S0–S10 na zamrożonym HGT. S10 wygrał i wszedł do FINAL."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from taskA.experiments.stage_c_stats.runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
