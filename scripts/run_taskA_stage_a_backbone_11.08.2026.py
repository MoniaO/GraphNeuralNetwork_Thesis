#!/usr/bin/env python3
"""CLI entry for Stage A backbone race — block 11.08.2026."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from taskA_final_large_grid_11_08_2026.stage_a_runner import main


if __name__ == "__main__":
    raise SystemExit(main())
