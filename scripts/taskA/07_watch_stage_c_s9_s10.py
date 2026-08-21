#!/usr/bin/env python3
"""07 — watchdog Stage C S9+S10 × 6 scenariuszy (opcjonalny)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOG = REPO / "outputs/taskA_final_large_grid_11.08.2026/logs"
RUNS = REPO / "outputs/taskA_final_large_grid_11.08.2026/stage_c/runs"
PIDF = LOG / "stage_c_s9_s10_6scen_11.08.2026.pid"
NOHUP = LOG / "stage_c_s9_s10_6scen_11.08.2026.nohup.log"
WATCH_PID = LOG / "stage_c_s9_s10_6scen_watchdog_11.08.2026.pid"
TOTAL = 36  # 2 variants × 6 scenarios × 3 seeds
POLL = 120
PATTERN = "05_run_stage_c_stats.py"
VARIANTS = ("S9_HCR_COMPACT36", "S10_HCR_FULL40")
SCENARIOS = (
    "clean",
    "hidden_confounder",
    "selection_bias",
    "no_overlap",
    "noisy_documentation",
    "multihospital",
)


def ok() -> int:
    n = 0
    if not RUNS.exists():
        return 0
    for sc in SCENARIOS:
        for v in VARIANTS:
            for seed in (20260721, 20260722, 20260723):
                p = RUNS / sc / "hd4" / v / f"seed{seed}" / "result_11.08.2026.json"
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
    # Confirm it is our multi runner (avoid stale PID reuse).
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode(
            errors="ignore"
        )
        return PATTERN in cmd and "--mode" in cmd
    except Exception:
        # macOS: no /proc — fall back to ps
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
    # Fallback: pgrep excluding the pgrep process itself
    try:
        out = subprocess.check_output(["pgrep", "-f", PATTERN], text=True).strip()
    except subprocess.CalledProcessError:
        return False
    for line in out.splitlines():
        try:
            pid = int(line.strip())
        except ValueError:
            continue
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
    logf.write(
        f"\n# watchdog restart {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
    )
    logf.flush()
    cmd = [
        str(REPO / ".venv/bin/python"),
        "-u",
        str(REPO / "scripts/taskA/05_run_stage_c_stats.py"),
        "--mode",
        "multi",
        "--all-scenarios",
        "--variant",
        "S9",
        "--variant",
        "S10",
        "--heads",
        "4",
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
        print(f"[watchdog-c-s9s10] ok={n}/{TOTAL} alive={a}", flush=True)
        if n >= TOTAL:
            print("COMPLETE", flush=True)
            raise SystemExit(0)
        if not a:
            start()
        time.sleep(POLL)
