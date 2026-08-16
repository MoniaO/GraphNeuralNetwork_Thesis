#!/usr/bin/env python3
"""Keep Stage A shared screen alive until full shared grid is done.

Target = count_shared_jobs(3) (currently 4 backbones × 24 × 3 = 288).
Restarts the detached runner if no Stage A runner is alive.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOG_ROOT = REPO / "outputs" / "taskA_final_large_grid_11.08.2026" / "logs"
RUNS = (
    REPO
    / "outputs"
    / "taskA_final_large_grid_11.08.2026"
    / "stage_a"
    / "runs"
    / "shared"
    / "clean"
)
PID_FILE = LOG_ROOT / "stage_a_shared_screen_11.08.2026.pid"
WATCH_PID = LOG_ROOT / "stage_a_watchdog_11.08.2026.pid"
NOHUP = LOG_ROOT / "stage_a_shared_screen_11.08.2026.nohup.log"
POLL_S = 120
RUNNER_PATTERN = "run_taskA_stage_a_backbone_11.08.2026.py"


def planned_total() -> int:
    sys.path.insert(0, str(REPO / "src"))
    from taskA_final_large_grid_11_08_2026.stage_a_grid import (  # noqa: WPS433
        count_shared_jobs,
    )

    return int(count_shared_jobs(3))


def count_ok() -> int:
    n = 0
    for p in RUNS.rglob("result_11.08.2026.json"):
        try:
            if json.loads(p.read_text(encoding="utf-8")).get("status") == "ok":
                n += 1
        except Exception:
            continue
    return n


def runner_alive() -> bool:
    try:
        subprocess.check_output(["pgrep", "-f", RUNNER_PATTERN], text=True)
        return True
    except subprocess.CalledProcessError:
        return False


def start_runner() -> int:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["GSN_PROJECT_ROOT"] = str(
        Path.home() / "Desktop" / "GSN Graphs dysertation 2026"
    )
    env["PYTHONPATH"] = str(REPO / "src")
    env["PYTHONUNBUFFERED"] = "1"
    logf = open(NOHUP, "a", encoding="utf-8")
    logf.write(
        f"\n# watchdog restart {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
    )
    logf.flush()
    proc = subprocess.Popen(
        [
            str(REPO / ".venv" / "bin" / "python"),
            "-u",
            str(REPO / "scripts" / "run_taskA_stage_a_backbone_11.08.2026.py"),
            "--mode",
            "shared_screen",
            "--scenario",
            "clean",
        ],
        cwd=str(REPO),
        env=env,
        stdout=logf,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    PID_FILE.write_text(str(proc.pid), encoding="utf-8")
    print(f"[watchdog] started runner pid={proc.pid}", flush=True)
    return proc.pid


def main() -> int:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    WATCH_PID.write_text(str(os.getpid()), encoding="utf-8")
    total = planned_total()
    print(
        f"[watchdog] pid={os.getpid()} poll={POLL_S}s target_ok={total}",
        flush=True,
    )
    while True:
        total = planned_total()
        ok = count_ok()
        alive = runner_alive()
        print(f"[watchdog] ok={ok}/{total} runner_alive={alive}", flush=True)
        if ok >= total:
            print("[watchdog] COMPLETE", flush=True)
            return 0
        if not alive:
            start_runner()
        time.sleep(POLL_S)


if __name__ == "__main__":
    raise SystemExit(main())
