#!/usr/bin/env python3
"""Wave 5A.1 — Common-Query and Coverage Audit.

No new training. Rebuilds paths with the fixed train-edge contract and
scores every frozen query (TPM=0 when γ* ∉ Ω; never drop from denominator).

Variants:
  uniform | hgt_finite | hcr2_finite
  edgewise_shuffle_global | edgewise_shuffle_matched | patient_shuffle_refit
"""

from __future__ import annotations

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
EVID_PS = OUT / "evidence_patient_shuffle"
QUERIES = OUT / "queries" / "wave5_frozen_queries.csv"
EVAL = OUT / "eval_5a1"
LOG = OUT / "wave5a1_runner.log"

SEEDS = [20260721, 20260722, 20260723, 20260724, 20260725]
VARIANTS = [
    ("uniform", ROOT / "configs/wnerw/uniform.yaml", "base"),
    ("hgt_finite", ROOT / "configs/wnerw/hgt_finite.yaml", "base"),
    ("hcr2_finite", ROOT / "configs/wnerw/hcr2_finite.yaml", "base"),
    ("edgewise_shuffle_global", ROOT / "configs/wnerw/edgewise_shuffle_global.yaml", "base"),
    ("edgewise_shuffle_matched", ROOT / "configs/wnerw/edgewise_shuffle_matched.yaml", "base"),
    ("patient_shuffle_refit", ROOT / "configs/wnerw/patient_shuffle_refit.yaml", "patient_shuffle"),
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


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(ROOT), env=ENV, capture_output=True, text=True)


def main() -> None:
    EVAL.mkdir(parents=True, exist_ok=True)
    log("=" * 60)
    log("WAVE 5A.1 — COMMON-QUERY AND COVERAGE AUDIT")
    log(f"Started: {datetime.now()}")
    log(f"Variants: {[v for v, _, _ in VARIANTS]}")
    log(f"Seeds: {SEEDS}")
    log("=" * 60)

    if not QUERIES.exists():
        log("Building frozen query registry...")
        proc = run([PYTHON, "scripts/wave5/build_frozen_queries.py", "--out", str(QUERIES)])
        with LOG.open("a") as lf:
            lf.write(proc.stdout or "")
            lf.write(proc.stderr or "")
        if proc.returncode != 0:
            log(f"FAIL build queries: {proc.stderr[-1000:]}")
            raise SystemExit(proc.returncode)
        log((proc.stdout or "").strip())

    # Patient-shuffle evidence (strong control).
    missing_ps = [
        s for s in SEEDS if not (EVID_PS / f"seed{s}_patient_shuffle.csv").exists()
    ]
    if missing_ps:
        log(f"Exporting patient_shuffle_refit evidence seeds={missing_ps} ...")
        proc = run(
            [
                PYTHON,
                "scripts/wave5/export_patient_shuffle_evidence.py",
                "--base-evidence-dir",
                str(EVID),
                "--out-dir",
                str(EVID_PS),
                "--seeds",
                *[str(s) for s in missing_ps],
            ]
        )
        with LOG.open("a") as lf:
            lf.write(proc.stdout or "")
            lf.write(proc.stderr or "")
        if proc.returncode != 0:
            log(f"FAIL patient_shuffle export: {(proc.stderr or '')[-1500:]}")
            raise SystemExit(proc.returncode)
        log("patient_shuffle export done.")

    for seed in SEEDS:
        base_evid = EVID / f"seed{seed}_edges.csv"
        if not base_evid.exists():
            base_evid = EVID / f"seed{seed}_edges.parquet"
        ps_evid = EVID_PS / f"seed{seed}_patient_shuffle.csv"
        for name, cfg, evid_kind in VARIANTS:
            evid = ps_evid if evid_kind == "patient_shuffle" else base_evid
            if not evid.exists():
                log(f"MISSING evidence {evid} for {name} seed={seed}")
                continue
            out_csv = EVAL / f"query_metrics_seed{seed}_{name}.csv"
            if out_csv.exists():
                log(f"SKIP {name} seed={seed}")
                continue
            log(f"===== AUDIT {name} seed={seed} =====")
            proc = run(
                [
                    PYTHON,
                    "scripts/wave5/run_wave5_coverage_audit.py",
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
                    "--out",
                    str(out_csv),
                ]
            )
            with LOG.open("a") as lf:
                lf.write(proc.stdout or "")
                lf.write(proc.stderr or "")
            if proc.returncode != 0:
                log(f"FAIL {name} seed={seed} rc={proc.returncode}")
                log((proc.stderr or "")[-800:])
            else:
                log((proc.stdout or "").strip().splitlines()[-1] if proc.stdout else "ok")

    log("===== SUMMARIZE PANEL A/B =====")
    proc = run(
        [
            PYTHON,
            "scripts/wave5/summarize_wave5a1.py",
            "--eval-dir",
            str(EVAL),
            "--out-dir",
            str(OUT),
        ]
    )
    with LOG.open("a") as lf:
        lf.write(proc.stdout or "")
        lf.write(proc.stderr or "")
    print(proc.stdout or "")
    if proc.returncode != 0:
        log(f"FAIL summarize: {(proc.stderr or '')[-800:]}")
        raise SystemExit(proc.returncode)

    (OUT / "WAVE5A1_PROTOCOL.txt").write_text(
        "Wave 5A.1 — Common-Query and Coverage Audit\n"
        "TPM_q = P(γ*_q|s,e) if γ* in Ω else 0; queries never dropped.\n"
        "Graph contract: train edges always kept; topological_allowed only for predicted.\n"
        "Panel A: all frozen queries (coverage + unconditional TPM/Hit@K).\n"
        "Panel B: common reachable set (paired TPM/MRR/Hit@K).\n"
        "Controls: edgewise_shuffle_global | edgewise_shuffle_matched | patient_shuffle_refit\n"
        f"Finished: {datetime.now()}\n"
    )
    log(f"WAVE5A1 finished {datetime.now()} → {OUT}")


if __name__ == "__main__":
    main()
