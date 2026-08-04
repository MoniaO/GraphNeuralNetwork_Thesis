"""Resolve paths to the GSN dissertation data tree on disk.

Layout (v3 is current; v2.2 lives under '-1. old v2_2'):

  GSN Graphs dysertation 2026/
    2 v3. Data/dataset_v3/     ← Task A training data
    -1. old v2_2/2. Data/      ← legacy v2.2 only

Set GSN_PROJECT_ROOT if your folder is not on Desktop, e.g.:
  export GSN_PROJECT_ROOT="$HOME/Desktop/GSN Graphs dysertation 2026"
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_GSN_ROOT = Path.home() / "Desktop" / "GSN Graphs dysertation 2026"


def gsn_root() -> Path:
    env = os.environ.get("GSN_PROJECT_ROOT")
    if env:
        root = Path(env).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"GSN_PROJECT_ROOT does not exist: {root}")
        return root

    if DEFAULT_GSN_ROOT.is_dir():
        return DEFAULT_GSN_ROOT

    for base in [Path.cwd(), *Path.cwd().parents]:
        if (base / "2 v3. Data").is_dir():
            return base

    raise FileNotFoundError(
        "Could not find GSN project root. Set GSN_PROJECT_ROOT to the folder that "
        "contains '2 v3. Data'."
    )


def v3_data_dir() -> Path:
    return gsn_root() / "2 v3. Data" / "dataset_v3"


def v3_code_dir() -> Path:
    return gsn_root() / "2 v3. Data" / "code"


def v3_splits_dir() -> Path:
    return gsn_root() / "2 v3. Data" / "splits"


def v22_data_dir() -> Path:
    """Legacy v2.2 (moved under -1. old v2_2)."""
    root = gsn_root()
    candidates = [
        root / "-1. old v2_2" / "2. Data" / "dataset_v2_2",
        root / "2. Data" / "dataset_v2_2",  # fallback if not yet moved
    ]
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError(
        "dataset_v2_2 not found under '-1. old v2_2/2. Data' or '2. Data'"
    )


def v22_code_dir() -> Path:
    root = gsn_root()
    candidates = [
        root / "-1. old v2_2" / "2. Data" / "code",
        root / "2. Data" / "code",
    ]
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError("v2.2 code folder not found")
