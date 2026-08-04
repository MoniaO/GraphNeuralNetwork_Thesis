#!/usr/bin/env python3
"""Wave 5B — HCR-to-Path Bridge matrix (D1 → q_uv + optional gate factors).

Exports D1 bridge evidence (if missing), then runs:
  d1_log_prob | d1_delta_logit | d1_nested | d1_log_prob_gates | d1_nested_gates | shuffled_d1
× seeds 20260721–23 (frozen Wave-4D D1 checkpoints).
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
EVID = OUT / "evidence_d1"
QUERIES = OUT / "queries" / "frozen_path_queries.csv"
PATHS = OUT / "paths_5b"
EVAL = OUT / "eval_5b"
LOG = OUT / "wave5b_runner.log"

SEEDS = [20260721, 20260722, 20260723]
VARIANTS = [
    ("d1_log_prob", ROOT / "configs/wnerw/d1_log_prob.yaml"),
    ("d1_delta_logit", ROOT / "configs/wnerw/d1_delta_logit.yaml"),
    ("d1_nested", ROOT / "configs/wnerw/d1_nested.yaml"),
    ("d1_log_prob_gates", ROOT / "configs/wnerw/d1_log_prob_gates.yaml"),
    ("d1_nested_gates", ROOT / "configs/wnerw/d1_nested_gates.yaml"),
    ("shuffled_d1", ROOT / "configs/wnerw/shuffled_d1.yaml"),
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
    EVID.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("WAVE 5B — HCR-TO-PATH BRIDGE (D1 → q_uv)")
    log(f"Started: {datetime.now()}")
    log(f"Variants: {[v for v, _ in VARIANTS]}")
    log(f"Seeds: {SEEDS}")
    log("=" * 60)

    # Export D1 bridge evidence if any seed missing.
    missing = [s for s in SEEDS if not (EVID / f"seed{s}_d1_bridge.csv").exists()]
    if missing:
        log(f"Exporting D1 bridge evidence for seeds={missing} ...")
        proc = subprocess.run(
            [
                PYTHON,
                "scripts/wave5/export_d1_bridge_evidence.py",
                "--out-dir",
                str(EVID),
                "--seeds",
                *[str(s) for s in missing],
                "--device",
                "cpu",
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
            log(f"FAIL export rc={proc.returncode}")
            log(proc.stderr[-2000:] if proc.stderr else "")
            raise SystemExit(proc.returncode)
        log("Export done.")

    if not QUERIES.exists():
        log("Building frozen path queries...")
        subprocess.run(
            [PYTHON, "scripts/wave5/build_path_queries.py", "--out", str(QUERIES)],
            cwd=str(ROOT),
            env=ENV,
            check=True,
        )

    for seed in SEEDS:
        evid = EVID / f"seed{seed}_d1_bridge.csv"
        if not evid.exists():
            log(f"MISSING evidence for seed={seed}: {evid}")
            continue
        for name, cfg_path in VARIANTS:
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
                        str(cfg_path),
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
                    log(proc.stderr[-800:] if proc.stderr else "")
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
            str(OUT / "WAVE5B_summary.csv"),
        ],
        cwd=str(ROOT),
        env=ENV,
        check=False,
    )
    (OUT / "WAVE5B_PROTOCOL.txt").write_text(
        "Wave 5B — HCR-to-Path Bridge\n"
        "Frozen context selector: structural_latent_pairwise (Wave 4D D1)\n"
        "Hide: parent_a (A→G held out of G_train)\n"
        "Bridge: q_AG = log P_D1(A→G); optional Δlogit / nested vs HGT\n"
        "Gate factors: population mean log P_D1(parent→gate) on entering gate\n"
        "Variants: d1_log_prob | d1_delta_logit | d1_nested | "
        "d1_log_prob_gates | d1_nested_gates | shuffled_d1\n"
        "Seeds: 20260721–23 (D1 checkpoints)\n"
        f"Finished: {datetime.now()}\n"
    )
    freeze = OUT / "HCR_WNERW_FREEZE.txt"
    if freeze.exists():
        text = freeze.read_text()
        if "Wave 5B" not in text or "COMPLETE" not in text:
            freeze.write_text(
                text.rstrip()
                + "\n\nWave 5B (complete):\n"
                + "  Bridge evidence: outputs/wave5/evidence_d1/\n"
                + "  Paths/eval: outputs/wave5/paths_5b/, eval_5b/\n"
                + "  Summary: outputs/wave5/WAVE5B_summary.csv\n"
            )
    log(f"WAVE5B finished {datetime.now()} → {OUT}")


if __name__ == "__main__":
    main()
