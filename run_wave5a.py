#!/usr/bin/env python3
"""Wave 5A — Finite Path Engine matrix (population-level WNERW).

Variants × 5 frozen edge-evidence seeds.
Uses pre-exported ``outputs/wave5/evidence/seed*_edges.csv``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "outputs/wave5"
EVID = OUT / "evidence"
QUERIES = OUT / "queries" / "frozen_path_queries.csv"
PATHS = OUT / "paths"
EVAL = OUT / "eval"
LOG = OUT / "wave5a_runner.log"

SEEDS = [20260721, 20260722, 20260723, 20260724, 20260725]
VARIANTS = [
    ("uniform", ROOT / "configs/wnerw/uniform.yaml"),
    ("hgt_finite", ROOT / "configs/wnerw/hgt_finite.yaml"),
    ("hcr2_finite", ROOT / "configs/wnerw/hcr2_finite.yaml"),
    ("shuffled_hcr", ROOT / "configs/wnerw/shuffled_hcr.yaml"),
]

PYTHON = str(ROOT / ".venv" / "bin" / "python")
if not Path(PYTHON).exists():
    PYTHON = sys.executable

ENV = os.environ.copy()
ENV["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT}:{ENV.get('PYTHONPATH', '')}"
ENV["PYTHONUNBUFFERED"] = "1"
ENV.setdefault(
    "GSN_PROJECT_ROOT",
    str(Path.home() / "Desktop" / "GSN Graphs dysertation 2026"),
)


def log(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = msg if msg.endswith("\n") else msg + "\n"
    with LOG.open("a") as f:
        f.write(line)
    print(line, end="", flush=True)


def main() -> None:
    PATHS.mkdir(parents=True, exist_ok=True)
    EVAL.mkdir(parents=True, exist_ok=True)
    log("=" * 60)
    log("WAVE 5A — FINITE PATH ENGINE")
    log(f"Started: {datetime.now()}")
    log(f"Variants: {[v for v, _ in VARIANTS]}")
    log(f"Seeds: {SEEDS}")
    log("=" * 60)

    if not QUERIES.exists():
        log("Building frozen path queries...")
        subprocess.run(
            [PYTHON, "scripts/wave5/build_path_queries.py", "--out", str(QUERIES)],
            cwd=str(ROOT),
            env=ENV,
            check=True,
        )

    for seed in SEEDS:
        evid = EVID / f"seed{seed}_edges.csv"
        if not evid.exists():
            evid = EVID / f"seed{seed}_edges.parquet"
        if not evid.exists():
            log(f"MISSING evidence for seed={seed}: {evid}")
            continue
        for name, cfg in VARIANTS:
            summary_path = PATHS / f"summary_seed{seed}_{name}.json"
            if summary_path.exists() and (PATHS / f"top_paths_seed{seed}_{name}.csv").exists():
                log(f"SKIP run {name} seed={seed}")
            else:
                log(f"===== RUN {name} seed={seed} =====")
                proc = subprocess.run(
                    [
                        PYTHON,
                        "scripts/wave5/run_wave5.py",
                        "--evidence",
                        str(evid),
                        "--queries",
                        str(QUERIES),
                        "--config",
                        str(cfg),
                        "--seed",
                        str(seed),
                        "--variant",
                        name,
                        "--out-dir",
                        str(PATHS),
                    ],
                    cwd=str(ROOT),
                    env=ENV,
                    capture_output=True,
                    text=True,
                )
                with LOG.open("a") as lf:
                    lf.write(proc.stdout or "")
                    lf.write(proc.stderr or "")
                if proc.returncode != 0:
                    log(f"FAIL run {name} seed={seed} rc={proc.returncode}")
                    log(proc.stderr[-500:] if proc.stderr else "")
                    continue
                log((proc.stdout or "").strip().splitlines()[-1] if proc.stdout else "ok")

            paths_csv = PATHS / f"top_paths_seed{seed}_{name}.csv"
            eval_csv = EVAL / f"eval_seed{seed}_{name}.csv"
            if eval_csv.exists():
                log(f"SKIP eval {name} seed={seed}")
                continue
            if not paths_csv.exists() or paths_csv.stat().st_size < 10:
                log(f"WARN empty paths {name} seed={seed}")
                continue
            log(f"===== EVAL {name} seed={seed} =====")
            ev = subprocess.run(
                [
                    PYTHON,
                    "scripts/wave5/evaluate_wave5.py",
                    "--paths",
                    str(paths_csv),
                    "--out",
                    str(eval_csv),
                ],
                cwd=str(ROOT),
                env=ENV,
                capture_output=True,
                text=True,
            )
            with LOG.open("a") as lf:
                lf.write(ev.stdout or "")
                lf.write(ev.stderr or "")
            if ev.returncode != 0:
                log(f"FAIL eval {name} seed={seed}")
                log(ev.stderr[-500:] if ev.stderr else "")
            else:
                log((ev.stdout or "").strip())

    log("===== SUMMARIZE =====")
    subprocess.run(
        [
            PYTHON,
            "scripts/wave5/summarize_wave5.py",
            "--eval-dir",
            str(EVAL),
            "--out",
            str(OUT / "WAVE5A_summary.csv"),
        ],
        cwd=str(ROOT),
        env=ENV,
    )
    (OUT / "WAVE5A_PROTOCOL.txt").write_text(
        "Wave 5A — Finite Path Engine (population-level WNERW)\n"
        "Frozen encoder: HGT L1 H8 hidden=64\n"
        "Frozen pair HCR: binary_compact (Wave 3/4A)\n"
        "Wave 4D freeze: structural_latent_pairwise = deployable context selector\n"
        "  (used in Wave 5B motif/path bridge, not as unrestricted scan)\n"
        "Variants: uniform | hgt_finite | hcr2_finite | shuffled_hcr\n"
        "Seeds: 20260721–25 (edge-model seeds; WNERW itself is deterministic)\n"
        "G_true used only in evaluate_wave5 (true-path mass / recall)\n"
        f"Finished: {datetime.now()}\n"
    )
    # Unpark marker
    parked = OUT.parent / "WAVE5_PARKED.txt"
    if parked.exists():
        parked.rename(OUT.parent / "WAVE5_UNPARKED.txt")
        (OUT.parent / "WAVE5_UNPARKED.txt").write_text(
            "Wave 5 UNPARKED after Wave 4D freeze.\n"
            "HCR evidence frozen: binary_compact (edge) + structural_latent_pairwise (context).\n"
            "Wave 5A finite-path matrix running / complete under outputs/wave5/.\n"
        )
    log(f"WAVE5A finished {datetime.now()} → {OUT}")


if __name__ == "__main__":
    main()
