#!/usr/bin/env python3
"""00 — coparent-Z registry (required by S10).

Writes `outputs/taskA/context_registry/edge_context_registry.csv`.
Run once, before Stage C / FINAL. Does not use G_true.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from taskA.features.context.coparent.runtime import load_cfg, load_seed_bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/taskA/features/context_registry.yaml")
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    out = ROOT / cfg["paths"]["output_dir"]
    out.mkdir(parents=True, exist_ok=True)

    frames = []
    adm = None
    for seed in cfg["seeds"]:
        bundle = load_seed_bundle(cfg, int(seed))
        frames.append(bundle["registry"])
        adm = bundle["admissible_df"]
        print(
            f"seed={seed} registry_rows={len(bundle['registry'])} "
            f"admissible={len(bundle['admissible'])}",
            flush=True,
        )

    reg = pd.concat(frames, ignore_index=True)
    reg.to_csv(out / "edge_context_registry.csv", index=False)
    assert "true_graph" not in set(reg["selector_source"].astype(str).unique())
    if adm is not None:
        adm.to_csv(out / "admissible_patient_features.csv", index=False)
    print(f"Wrote {out} ({len(reg)} registry rows)")


if __name__ == "__main__":
    main()
