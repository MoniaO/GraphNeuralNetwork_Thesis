#!/usr/bin/env python3
"""08 — FINAL 14.08: S10 × MLP/KAN × 6 scenariuszy × 5 seedów (60 jobów)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from taskA.experiments.final_14_08.runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
