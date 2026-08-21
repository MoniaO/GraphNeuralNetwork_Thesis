#!/usr/bin/env python3
"""06 — watchdog Stage C (opcjonalny, scenariusz clean, heads=4)."""
import json, os, subprocess, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
LOG = REPO / "outputs/taskA_final_large_grid_11.08.2026/logs"
RUNS = REPO / "outputs/taskA_final_large_grid_11.08.2026/stage_c/runs/clean/hd4"
PIDF = LOG / "stage_c_screen_clean_hd4_11.08.2026.pid"
NOHUP = LOG / "stage_c_screen_clean_hd4_11.08.2026.nohup.log"
TOTAL = 33
POLL = 120
PATTERN = "05_run_stage_c_stats.py"

def ok():
    n=0
    if not RUNS.exists(): return 0
    for p in RUNS.rglob("result_11.08.2026.json"):
        try:
            if json.loads(p.read_text()).get("status")=="ok": n+=1
        except Exception: pass
    return n

def alive():
    try:
        subprocess.check_output(["pgrep","-f",PATTERN])
        return True
    except subprocess.CalledProcessError:
        return False

def start():
    env=os.environ.copy()
    env["GSN_PROJECT_ROOT"]=str(Path.home()/"Desktop"/"GSN Graphs dysertation 2026")
    env["PYTHONPATH"]=str(REPO/"src"); env["PYTHONUNBUFFERED"]="1"
    LOG.mkdir(parents=True, exist_ok=True)
    logf=open(NOHUP,"a"); logf.write(f"\n# watchdog restart {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"); logf.flush()
    p=subprocess.Popen([str(REPO/".venv/bin/python"),"-u",str(REPO/"scripts/taskA/05_run_stage_c_stats.py"),"--mode","screen","--scenario","clean","--heads","4"], cwd=str(REPO), env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    PIDF.write_text(str(p.pid)); print("started", p.pid, flush=True)

if __name__=="__main__":
    (LOG/"stage_c_watchdog_11.08.2026.pid").write_text(str(os.getpid()))
    while True:
        n=ok(); a=alive(); print(f"[watchdog-c] ok={n}/{TOTAL} alive={a}", flush=True)
        if n>=TOTAL: print("COMPLETE", flush=True); raise SystemExit(0)
        if not a: start()
        time.sleep(POLL)
