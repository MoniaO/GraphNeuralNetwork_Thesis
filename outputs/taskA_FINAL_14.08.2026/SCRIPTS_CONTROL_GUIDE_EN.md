# FINAL scripts — what they do and what to control

Numbered entrypoints: [`scripts/taskA/README.md`](../../scripts/taskA/README.md) (**00 → 01 → 05 → 08 → 10 → 11**).
Library code: `src/taskA/` (data / features / models / training / experiments).

Selection: **valid AUPRC** only. Test is **sealed**. Candidate seed: **`20260722`**.

```bash
export GSN_PROJECT_ROOT="$HOME/Desktop/GSN Graphs dysertation 2026"
export PYTHONPATH=src
```

## FINAL 14.08 CLI

| Script | Role |
|---|---|
| `scripts/taskA/08_run_final.py` | 60-job grid (MLP vs KAN × 6 scenarios × 5 seeds) |
| `scripts/taskA/09_watch_final.py` | watchdog until 60/60 |
| `scripts/taskA/10_plot_learning_curves.py` | CSV + PNG from logs |
| `scripts/taskA/11_eval_edge_pathway.py` | edge × pathway / rare |

```bash
.venv/bin/python scripts/taskA/08_run_final.py --mode count
.venv/bin/python scripts/taskA/08_run_final.py --mode smoke --encoder mlp
.venv/bin/python scripts/taskA/08_run_final.py --mode full
```

`--mode` values: `count` / `smoke` / `full`. After a smoke run, delete those run dirs before `--mode full`, or skip-ok will keep the 1-epoch checkpoints.

Freeze constants live in `src/taskA/experiments/final_14_08/__init__.py` (`FROZEN`, `FINAL_SEEDS`, `SCENARIOS`, `CANDIDATE_SEED`). Do not edit them to reproduce the 14.08 table.

## 11.08 campaign (already frozen)

| Script | Role |
|---|---|
| `scripts/taskA/00_build_context_registry.py` | coparent-Z registry for S10 |
| `scripts/taskA/01_run_stage_a_backbone.py` | encoder race (HGT won) |
| `scripts/taskA/05_run_stage_c_stats.py` | S0–S10 (S10 won) |

Outputs stay under `outputs/taskA_final_large_grid_11.08.2026/` — do not overwrite them with FINAL jobs.
