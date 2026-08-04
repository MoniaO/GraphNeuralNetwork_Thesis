#!/usr/bin/env python3
"""Thin entrypoint: print resolved GSN paths (same as notebooks/gsn_paths.py)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "notebooks"))

from gsn_paths import gsn_root, v3_code_dir, v3_data_dir, v3_splits_dir  # noqa: E402

if __name__ == "__main__":
    print(f"GSN project root: {gsn_root()}")
    print(f"v3 dataset:      {v3_data_dir()}")
    print(f"v3 code:         {v3_code_dir()}")
    print(f"v3 splits:       {v3_splits_dir()}")
