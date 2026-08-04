#!/usr/bin/env python3
"""Rebuild E3 exploratory + main summaries from an existing raw CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.taskA_per_edge_type_summaries import (  # noqa: E402
    PRIORITY_EDGE_TYPES,
    write_per_edge_type_summaries,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "taskA_per_edge_type.csv",
    )
    parser.add_argument("--split", default="valid")
    args = parser.parse_args()

    paths = write_per_edge_type_summaries(args.raw, split=args.split)
    print("Priority relations:", ", ".join(PRIORITY_EDGE_TYPES))
    for name, path in paths.items():
        print(f"{name}: {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
