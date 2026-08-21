# FINAL scripts — what they do and what to control

Scripts live in **`scripts/taskA/`** (numbered 00–11). See [`scripts/README.md`](../../scripts/README.md).
Old names in this file map 1:1 (`08_run_final.py` → `scripts/taskA/08_run_final.py`).
Library code is under **`src/taskA/`** (data / features / models/encoder / models/decoder / training / experiments).

**Language:** English (operator / thesis review guide)  
**Scope:** Task A FINAL 14.08.2026 and the 11.08 campaign scripts that produced the freeze  
**Out of scope:** Wave11 Wave7C / MOOC proxy scripts (different stacks)

Selection rule everywhere below: **valid AUPRC** only. Test metrics are **sealed** (report only).  
Fixed candidate seed: **`20260722`** (must not follow the training seed).

Environment (typical):

```bash
export GSN_PROJECT_ROOT="$HOME/Desktop/GSN Graphs dysertation 2026"
export PYTHONPATH=src
# Prefer: .venv/bin/python
```

---

## A. FINAL 14.08.2026 (primary)

### 1. `scripts/08_run_final.py`

**What it does**  
Thin CLI entry. Imports `taskA.experiments.final_14_08.runner.main` and launches the FINAL grid: frozen HGT + `fusion88_stat` + **S10_HCR_FULL40**, for encoder twins **MLP** and/or **KAN-shallow**, across 6 scenarios × 5 `FINAL_SEEDS`.

Each job calls `src/train_taskA.py` with Hydra overrides, writes under:

`outputs/taskA_FINAL_14.08.2026/runs/{mlp|kan_shallow}/{scenario}/S10_HCR_FULL40/seed*/`

Artifacts per run: `best_model.pt`, `train_14.08.2026.log`, `result_14.08.2026.json`, `launch_meta_14.08.2026.json`.  
Skips runs that already have `result_*.json` with `"status": "ok"`.

**How to control**

| Flag / knob | Effect |
|---|---|
| `--mode count` | Print planned job count; no training |
| `--mode smoke` | 1 epoch, 1 seed, first scenario; use to verify wiring |
| `--mode full` | Full grid (default encoders × scenarios × 5 seeds) |
| `--encoder mlp` / `--encoder kan_shallow` | Repeatable; default = both |
| `--scenario NAME` | Repeatable; default = all 6 GSN scenarios |
| `--max-jobs N` | Cap number of jobs |
| `--epochs` / `--patience` | Override training length / early stop |
| `--dry-run` | Build cmds / dirs only |

**Code knobs (not CLI)** — edit `src/taskA.experiments.final_14_08/__init__.py`:

- `FROZEN` — lr, layers, heads, dropout, epochs, patience, grad_clip  
- `FINAL_SEEDS`, `SCENARIOS`, `CANDIDATE_SEED`  
- `MODEL_NAME` / `BLOCK_ID`

**Code knobs** — edit `src/taskA.experiments.final_14_08/runner.py`:

- `get_variant("S10")` — switch to S9 only with a deliberate protocol change  
- W&B group / run_name / tags  
- `OUT_ROOT`  
- Hydra overrides (`stat_pair_encoder`, `stat_raw_dim`, device, …)

**Example**

```bash
.venv/bin/python scripts/08_run_final.py --mode count
.venv/bin/python scripts/08_run_final.py --mode smoke --encoder mlp
.venv/bin/python scripts/08_run_final.py --mode full
```

**Warning:** After smoke (1-epoch), delete those run dirs before `--mode full`, or skip-ok will keep the smoke checkpoints.

---

### 2. `scripts/watch_FINAL_14.08.2026.py`

**What it does**  
Watchdog for the FINAL grid. Polls completed `ok` results (target **60**). If the runner PID is dead, restarts:

`08_run_final.py --mode full`

Uses PID-file liveness (not a naive self-matching `pgrep -f`).

**How to control**

| Constant in script | Effect |
|---|---|
| `TOTAL` | Expected ok count (60) |
| `POLL` | Seconds between checks (default 180) |
| `PIDF` / `NOHUP` / `LOG` | Log and PID paths under `outputs/taskA_FINAL_14.08.2026/logs/` |

No CLI flags. Start detached with `start_new_session=True` / `nohup`.

**Example**

```bash
.venv/bin/python -u scripts/watch_FINAL_14.08.2026.py
```

---

### 3. `scripts/plot_taska_learning_curves_14.08.2026.py`

**What it does**  
Parses train logs (`Epoch … \| optim … \| train AUPRC … \| valid AUPRC …`) and writes:

- per-run PNG (loss, train/valid AUPRC, valid AUC, train−valid gap)  
- per-run epoch CSV  
- `all_epoch_histories.csv`  
- `mean_valid_auprc_by_group.png`  
- `LEARNING_DIAGNOSTICS_14.08.2026.md`  

Sources: 11.08 Stage C S10 logs and/or FINAL 14.08 logs.

**How to control**

| Flag | Effect |
|---|---|
| `--source 11.08` | Only Stage C S10 (11.08) logs |
| `--source final` | Only FINAL 14.08 logs |
| `--source both` | Default; both trees |

Output root: `outputs/taskA_FINAL_14.08.2026/learning_curves/`.

**Example**

```bash
.venv/bin/python scripts/plot_taska_learning_curves_14.08.2026.py --source both
```

---

### 4. `scripts/eval_taska_stage_c_edge_pathway_report.py`

**What it does**  
Eval-only (no retrain). Loads a checkpoint, rebuilds data + Stage C S10 attach, runs forward pass, dumps:

- `predictions_{valid,test}_*.csv` — source, target, label, probability  
- joins audited edges → `clinical_pathway`, `edge_type`, `dag_motif`  
- pathway / edge_type / motif aggregate metrics  
- rare / mid / common prevalence buckets  
- `rare_pathway_table.md`, `ALL_SUMMARIES_*.json`  

**How to control**

| Flag | Effect |
|---|---|
| `--source 11.08` | Evaluate existing Stage C S10 ckpts |
| `--source final` | Evaluate FINAL `mlp` + `kan_shallow` ckpts |
| `--source both` | Both |
| `--encoder mlp\|kan_shallow` | Used when reconstructing FINAL overrides |
| `--max-runs N` | Cap number of ckpts |

Requires `GSN_PROJECT_ROOT` and audited edges CSV.  
Requires Wave5C registry for Stage C attach (same as training).

**Known caveat:** early 11.08 pathway join had **low name coverage** on some candidates (~0.25). Treat rare-pathway tables cautiously until join coverage is re-checked / fixed.

**Example**

```bash
.venv/bin/python scripts/eval_taska_stage_c_edge_pathway_report.py --source final
```

---

## B. Stage A — backbone race (11.08 freeze)

### 5. `scripts/01_run_stage_a_backbone.py`

**What it does**  
CLI → `stage_a_runner.main`. Shared-screen backbone race on `clean` with `fusion88` and **`g_stat = zeros(24)`** (no HCR / no Stage C stats). Produced the HGT freeze used by FINAL.

**How to control**

| Flag | Effect |
|---|---|
| `--mode count` | Job count |
| `--mode smoke` | Short smoke |
| `--mode shared_screen` | Full shared grid |
| `--mode refine_hgt_heads` | Heads 4 vs 8 refine |
| `--scenario` | Usually `clean` |
| `--max-jobs` / `--epochs` / `--patience` / `--dry-run` | As usual |

Grid definitions live in `src/taskA_final_large_grid_11_08_2026/stage_a_grid.py` (backbones, hypers).

**Do not** overwrite frozen Stage A artifacts casually; FINAL must keep the frozen HGT Top-1.

---

### 6. `scripts/watch_stage_a_shared_screen_11.08.2026.py`

**What it does**  
Watchdog for Stage A shared screen (historical; 288-job class run). Restarts the Stage A runner if it dies.

**Control:** `TOTAL`, poll interval, log paths inside the script.

---

### 7. `scripts/summarize_stage_a_backbone_11.08.2026.py`

**What it does**  
Aggregates Stage A `result_*.json` → ranking / decision tables (mean **valid AUPRC**; test sealed). Writes MD/CSV under `outputs/taskA_final_large_grid_11.08.2026/stage_a/`.

**How to control**

| Flag | Effect |
|---|---|
| `--scenario` | Which scenario tree to summarize (default `clean`) |

---

### 8. `scripts/upload_stage_a_curves_to_wandb_11.08.2026.py`

**What it does**  
Uploads / backfills Stage A training curves to Weights & Biases (historical ops helper).

**Control:** CLI args in that script (scenario / paths); W&B entity/project from env/login.

---

## C. Stage C — stats screen (11.08)

### 9. `scripts/05_run_stage_c_stats.py`

**What it does**  
CLI → `stage_c_runner.main`. Runs statistical variants **S0–S10** on the **frozen** HGT (heads=4 default) with `fusion88_stat`. Used for clean S0–S10 screen and later S9/S10 × 6 scenarios (`--mode multi`).

**How to control**

| Flag | Effect |
|---|---|
| `--mode count` | Job count |
| `--mode smoke` | Short smoke (default variants S0/S1 unless set) |
| `--mode screen` | Full variant × seed screen for one scenario |
| `--mode multi` | Loop scenarios (use with `--all-scenarios`) |
| `--all-scenarios` | All 6 GSN scenarios |
| `--variant S10` | Repeatable; e.g. S9 S10 |
| `--scenario` | Single scenario when not all |
| `--heads 4\|8` | Default 4 (won refine) |
| `--max-jobs` / `--epochs` / `--patience` / `--dry-run` | As usual |

Outputs: `outputs/taskA_final_large_grid_11.08.2026/stage_c/`.

---

### 10. `scripts/watch_stage_c_screen_11.08.2026.py`

**What it does**  
Watchdog for Stage C clean S0–S10 screen (33 jobs). Historical.

**Control:** `TOTAL`, paths, poll inside script.

---

### 11. `scripts/watch_stage_c_s9_s10_6scen_11.08.2026.py`

**What it does**  
Watchdog for S9+S10 × 6 scenarios × 3 seeds (36 jobs). Restarts multi-mode Stage C runner.

**Control:** `TOTAL=36`, variant/scenario lists, PID liveness helpers in script.

---

## D. Library code called by the scripts (not under `scripts/`, but operators edit these)

| Module | Role | What to control |
|---|---|---|
| `src/taskA/experiments/final_14_08/__init__.py` | FINAL constants / freeze | seeds, hypers, model name |
| `src/taskA/experiments/final_14_08/runner.py` | FINAL job launcher | overrides, OUT_ROOT, S10, encoders |
| `src/taskA/training/train.py` | Single training job | selection metric, early stop, W&B, Stage C attach |
| `src/taskA/models/decoder/fusion88.py` | Graph branch + fusion 88 | `GRAPH_MID`, `GRAPH_OUT`, `STAT_DIM` — freeze for 14.08 |
| `src/taskA/models/decoder/pair_encoder.py` | MLP / KAN pair encoders | `stat_pair_encoder`, spline_l1 |
| `src/taskA/features/variants.py` | S0–S10 registry | raw dims / new variants |
| `src/taskA/features/compute.py` | Feature computation | FULL40 slices, classical stats |
| `src/taskA/features/attach.py` | Train-only fit + role masks | registry path, mask rules |
| `src/taskA/models/link_predictor.py` | HGT + decoder wiring | encoder via Hydra |
| `src/taskA/models/encoder/hgt.py` | HGT layers | hidden / layers / heads |

---

## E. Recommended operator order

1. **Read** `outputs/taskA_FINAL_14.08.2026/PROTOCOL_14.08.2026.md`  
2. **Results** `SUMMARY_MACRO_14.08.2026.md`  
3. **Train (if re-run)** `watch_FINAL_14.08.2026.py` → `08_run_final.py`  
4. **Curves** `plot_taska_learning_curves_14.08.2026.py --source both`  
5. **Pathway** `eval_taska_stage_c_edge_pathway_report.py --source final`  
6. **Tests**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/taskA -q
```

---

## F. Git push — what belongs with FINAL

**Include (code + docs + light summaries):**

- `scripts/08_run_final.py`  
- `scripts/watch_FINAL_14.08.2026.py`  
- `scripts/plot_taska_learning_curves_14.08.2026.py`  
- `scripts/eval_taska_stage_c_edge_pathway_report.py`  
- related 11.08 Stage A/C CLIs if not already committed  
- `src/taskA.experiments.final_14_08/`  
- `src/taskA_final_large_grid_11_08_2026/` (incl. KAN in `stat_encoder.py`)  
- `tests/test_FINAL_14_08_2026_kan.py` (+ Stage C / fusion88 tests)  
- `outputs/taskA_FINAL_14.08.2026/*.md`, `MANIFEST_*.json`, `SUMMARY_MACRO_*.md`, `FINAL_SUMMARY_*.json`  
- this file: `SCRIPTS_CONTROL_GUIDE_EN.md`

**Usually exclude / gitignore:**

- `best_model.pt`  
- large `train_*.log` / `wandb/` run dirs  
- bulky PNG dumps if repo size is a concern (keep MD diagnostics)

---

## G. Hard rules (do not break in scripts)

1. Selection = **valid AUPRC** only.  
2. `candidate_seed=20260722` stays fixed.  
3. Do not retune HGT and S10 together after freeze.  
4. Do not overwrite `outputs/taskA_final_large_grid_11.08.2026/` with FINAL runs.  
5. Do not mix Wave11 Wave7C numbers into FINAL tables.

---

_Document prepared for the FINAL git push. Companion maps: `CODE_AND_SCRIPT_MAP_14.08.2026.md`, `REVIEW_ORDER_TODAY.md`._
