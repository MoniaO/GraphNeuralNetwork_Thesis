#!/usr/bin/env python3
"""CLI: Task A FINAL 14.08.2026 — S10 MLP + KAN twin grid."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from taskA_FINAL_14_08_2026.runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
