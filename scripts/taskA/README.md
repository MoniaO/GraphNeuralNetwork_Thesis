# Task A scripts — reproduction order

Everything lives in this folder. The number in the filename is the run order.

Scientific path (builds the result): **00 → 01 → 05 → 08 → 10 → 11**.
Watchdogs and summaries are optional.

| # | File | When | What it does |
|---|---|---|---|
| 00 | `00_build_context_registry.py` | once, before S10 | coparent-Z registry → `outputs/taskA/context_registry/` |
| 01 | `01_run_stage_a_backbone.py` | 11.08 campaign | encoder race (HGT won) |
| 02 | `02_watch_stage_a.py` | optional | Stage A watchdog |
| 03 | `03_summarize_stage_a.py` | after 01 | Stage A decision tables |
| 04 | `04_upload_stage_a_curves.py` | optional | Stage A curves → W&B |
| 05 | `05_run_stage_c_stats.py` | after HGT freeze | S0–S10 (S10 won) |
| 06 | `06_watch_stage_c.py` | optional | Stage C watchdog (clean) |
| 07 | `07_watch_stage_c_s9_s10.py` | optional | S9/S10 × 6 scenarios watchdog |
| 08 | `08_run_final.py` | after S10 freeze | FINAL 14.08: 6 × 5 × MLP/KAN |
| 09 | `09_watch_final.py` | optional | 60-job watchdog |
| 10 | `10_plot_learning_curves.py` | after 08 | CSV + PNG from logs |
| 11 | `11_eval_edge_pathway.py` | after 08 | predictions × pathway / rare |

```bash
export GSN_PROJECT_ROOT="$HOME/Desktop/GSN Graphs dysertation 2026"
export PYTHONPATH="$PWD/src:$PWD"

# reproduce FINAL (assumes registry from 00 and the 11.08 freeze)
PYTHONPATH=src .venv/bin/python scripts/taskA/08_run_final.py --mode count
PYTHONPATH=src .venv/bin/python scripts/taskA/09_watch_final.py
PYTHONPATH=src .venv/bin/python scripts/taskA/10_plot_learning_curves.py --source final
PYTHONPATH=src .venv/bin/python scripts/taskA/11_eval_edge_pathway.py
```

Model code is in `src/taskA/`. These files are entrypoints only.
