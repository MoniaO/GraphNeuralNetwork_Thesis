#!/usr/bin/env python3
"""01 — Stage A: encoder race (HGT won). 11.08 campaign."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from taskA.experiments.stage_a_backbone.runner import main


if __name__ == "__main__":
    raise SystemExit(main())
