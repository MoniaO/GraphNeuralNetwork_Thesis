#!/usr/bin/env python3
"""Export Wave 3B motif catalog from G_true."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from hcr.motifs import export_motif_catalog

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    day = date.today().isoformat()
    out = ROOT / "outputs" / f"wave3b_motifs_{day}"
    paths = export_motif_catalog(out)
    print((out / "wave3b_motif_catalog_summary.txt").read_text())
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
