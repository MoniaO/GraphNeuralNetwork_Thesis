# Task A — Final Large Train Grid (11.08.2026)

Separate experimental block. Does **not** mix graph-architecture tuning
with statistical features.

## Rule

| Stage | Goal | Statistics |
|---|---|---|
| **A** | choose backbone + graph hypers | `g_stat = 0` (24D zeros) |
| **B** | freeze the winning backbone | no changes |
| **C** | compare S0–S10 variants | evidence changes only here |

**Test is not used for selection** until architecture and representation are frozen.

## Shared data protocol

- `candidate_seed = 20260722` (**does not** follow the training seed)
- same patient/candidate splits, `G_train`, node features, loss, AUPRC
- screening seeds: `20260721–23`
- final confirmation: `20260721–25`
- scenarios: all 6 GSN v3
- Stage A: `hcr=none`, `decoder=fusion88`

## Run Stage A

```bash
export GSN_PROJECT_ROOT="$HOME/Desktop/GSN Graphs dysertation 2026"
export PYTHONPATH=src

# unit
.venv/bin/python -m pytest tests/taskA/test_decoder_fusion88.py -q

# smoke (2 epochs, 1 HGT config)
.venv/bin/python scripts/taskA/01_run_stage_a_backbone.py --mode smoke

# full shared screen on clean × 3 seeds
.venv/bin/python scripts/taskA/01_run_stage_a_backbone.py --mode shared_screen

# count jobs
.venv/bin/python scripts/taskA/01_run_stage_a_backbone.py --mode count
```

## Block layout

```
outputs/taskA_final_large_grid_11.08.2026/
  README_11.08.2026.md
  PROTOCOL_11.08.2026.md
  MANIFEST_11.08.2026.json
  stage_a/   stage_b/   stage_c/
  audit/     notebooks/ logs/

src/taskA/experiments/stage_a_backbone/
src/taskA/experiments/stage_c_stats/
configs/taskA/experiments/stage_a_backbone.yaml
scripts/taskA/01_run_stage_a_backbone.py
scripts/taskA/05_run_stage_c_stats.py
tests/taskA/
```

## Status

- [x] Block + protocol + manifest
- [x] Fusion88 decoder (`4d→128→64`, g_stat zeros)
- [x] `rgcn_matched` + type-specific projection
- [x] Stage A grid + runner (candidate_seed frozen)
- [x] Smoke Stage A (HGT / GAT / RGCN, 2 epochs)
- [x] Shared grid screen (HGT won; freeze heads=4)
- [x] Stage C S0–S10 (S10 won)
- [x] FINAL 14.08 lives in `outputs/taskA_FINAL_14.08.2026/`
