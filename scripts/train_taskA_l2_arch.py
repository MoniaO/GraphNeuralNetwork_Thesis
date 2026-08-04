#!/usr/bin/env python3
"""Thin pointer — L2 architecture audit trains via src/train_taskA.py.

Preferred entrypoints:
  ./run_TaskA_L2_ARCH_AUDIT.sh check
  ./run_TaskA_L2_ARCH_AUDIT.sh 0
  python scripts/run_taskA_arch_audit.py --stage 0

Direct example:
  python src/train_taskA.py \\
    model=TaskA_hgt_l2_arch \\
    hcr=structural_latent_pairwise \\
    experiment.wave=WAVE5D_ARCH_AUDIT \\
    data.candidate_seed=20260722 \\
    training.seed=20260722 \\
    training.epochs=300 \\
    training.early_stopping_patience=40
"""

from __future__ import annotations

import sys


def main() -> None:
    print(__doc__)
    print(
        "Use scripts/run_taskA_arch_audit.py or ./run_TaskA_L2_ARCH_AUDIT.sh",
        file=sys.stderr,
    )
    raise SystemExit(2)


if __name__ == "__main__":
    main()
