#!/usr/bin/env python3
"""Watchdog for FINAL 14.08.2026 (60 jobs: S10 mlp+kan × 6 × 5)."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOG = REPO / "outputs/taskA_FINAL_14.08.2026/logs"
RUNS = REPO / "outputs/taskA_FINAL_14.08.2026/runs"
PIDF = LOG / "FINAL_14.08.2026.pid"
NOHUP = LOG / "FINAL_14.08.2026.nohup.log"
WATCH_PID = LOG / "FINAL_14.08.2026_watchdog.pid"
TOTAL = 60
POLL = 180
PATTERN = "run_taskA_FINAL_14.08.2026.py"
ENCODERS = ("mlp", "kan_shallow")
SCENARIOS = (
    "clean",
    "hidden_confounder",
    "selection_bias",
    "no_overlap",
    "noisy_documentation",
    "multihospital",
)
SEEDS = (20260721, 20260722, 20260723, 20260724, 20260725)


def ok() -> int:
    n = 0
    for enc in ENCODERS:
        for sc in SCENARIOS:
            for seed in SEEDS:
                p = (
                    RUNS
                    / enc
                    / sc
                    / "S10_HCR_FULL40"
                    / f"seed{seed}"
                    / "result_14.08.2026.json"
                )
                if not p.exists():
                    continue
                try:
                    if json.loads(p.read_text(encoding="utf-8")).get("status") == "ok":
                        n += 1
                except Exception:
                    pass
    return n


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        out = subprocess.check_output(["ps", "-p", str(pid), "-o", "command="], text=True)
        return PATTERN in out
    except subprocess.CalledProcessError:
        return False


def alive() -> bool:
    if PIDF.exists():
        try:
            pid = int(PIDF.read_text(encoding="utf-8").strip())
        except Exception:
            pid = -1
        if _pid_alive(pid):
            return True
    return False


def start() -> None:
    env = os.environ.copy()
    env["GSN_PROJECT_ROOT"] = str(Path.home() / "Desktop" / "GSN Graphs dysertation 2026")
    env["PYTHONPATH"] = str(REPO / "src")
    env["PYTHONUNBUFFERED"] = "1"
    LOG.mkdir(parents=True, exist_ok=True)
    logf = open(NOHUP, "a", encoding="utf-8")
    logf.write(f"\n# watchdog restart {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
    logf.flush()
    cmd = [
        str(REPO / ".venv/bin/python"),
        "-u",
        str(REPO / "scripts/run_taskA_FINAL_14.08.2026.py"),
        "--mode",
        "full",
    ]
    p = subprocess.Popen(
        cmd,
        cwd=str(REPO),
        env=env,
        stdout=logf,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    PIDF.write_text(str(p.pid), encoding="utf-8")
    print("started", p.pid, flush=True)


if __name__ == "__main__":
    WATCH_PID.write_text(str(os.getpid()), encoding="utf-8")
    while True:
        n = ok()
        a = alive()
        print(f"[watchdog-FINAL] ok={n}/{TOTAL} alive={a}", flush=True)
        if n >= TOTAL:
            print("COMPLETE", flush=True)
            raise SystemExit(0)
        if not a:
            start()
        time.sleep(POLL)
