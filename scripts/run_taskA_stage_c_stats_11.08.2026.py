#!/usr/bin/env python3
"""CLI entry for Task A Final Large Grid Stage C (11.08.2026)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from taskA_final_large_grid_11_08_2026.stage_c_runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
