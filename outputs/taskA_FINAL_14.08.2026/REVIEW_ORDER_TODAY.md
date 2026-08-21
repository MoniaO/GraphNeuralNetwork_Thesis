# Code review order — FINAL 14.08.2026

Goal: read **data → encoder → decoder → training → experiments**, without old paths.
Verdict: MLP-stat (macro valid AUPRC **0.919**) vs KAN **0.908**.
Details: [`SUMMARY_MACRO_14.08.2026.md`](SUMMARY_MACRO_14.08.2026.md)

Naming: `src/taskA/<stage>/` (data, features, models/encoder, models/decoder, …).
Scripts: `scripts/taskA/00_…` … `11_…` in reproduction order.

---

## 0. Context (15 min)

1. [`00_README.md`](00_README.md) — what FINAL is
2. [`PROTOCOL_14.08.2026.md`](PROTOCOL_14.08.2026.md) — freeze / do-not-touch
3. [`src/taskA/README.md`](../../src/taskA/README.md) — package map
4. [`scripts/taskA/README.md`](../../scripts/taskA/README.md) — numbering 00–11

---

## 1. Architecture — read in this order

| # | File | Why |
|---:|---|---|
| 1 | `src/taskA/data/load_graph.py` | G_train + candidates |
| 2 | `src/taskA/features/variants.py` | S0–S10; FINAL = S10 40D |
| 3 | `src/taskA/features/attach.py` | train-only fit, `stat_raw` + masks |
| 4 | `src/taskA/models/encoder/hgt.py` | HGT L2 h32 |
| 5 | `src/taskA/models/decoder/fusion88.py` | graph 64 + stat 24 → 88 → logit |
| 6 | `src/taskA/models/decoder/pair_encoder.py` | **MLP vs KAN** |
| 7 | `src/taskA/models/link_predictor.py` | wires encoder to decoder |
| 8 | `src/taskA/training/train.py` | early stop on valid AUPRC |

---

## 2. FINAL runner (60 jobs)

| # | File | Why |
|---:|---|---|
| 1 | `src/taskA/experiments/final_14_08/__init__.py` | seeds, `FROZEN` |
| 2 | `src/taskA/experiments/final_14_08/runner.py` | Hydra overrides, skip-ok |
| 3 | `scripts/taskA/08_run_final.py` | CLI (`count` / `smoke` / `full`) |
| 4 | `scripts/taskA/09_watch_final.py` | watchdog |

---

## 3. 11.08 campaign (where the freeze came from)

| # | File | Why |
|---:|---|---|
| 1 | `scripts/taskA/01_run_stage_a_backbone.py` | encoder race |
| 2 | `src/taskA/experiments/stage_a_backbone/` | Stage A grid + runner |
| 3 | `scripts/taskA/05_run_stage_c_stats.py` | S0–S10 |
| 4 | `outputs/.../stage_a/STAGE_A_FREEZE_HGT_TOP1_11.08.2026.md` | HGT decision |
| 5 | `outputs/.../stage_c/STAGE_C_CLEAN_S0_S10_TABLE_11.08.2026.md` | S10 ranking |

---

## 4. Diagnostics

| # | File | Why |
|---:|---|---|
| 1 | `SUMMARY_MACRO_14.08.2026.md` | MLP vs KAN table |
| 2 | `scripts/taskA/10_plot_learning_curves.py` | curves from logs |
| 3 | `scripts/taskA/11_eval_edge_pathway.py` | edge × pathway |

---

## 5. Tests

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/taskA -q
```

| File | Covers |
|---|---|
| `tests/taskA/test_decoder_fusion88.py` | fusion 88 shapes |
| `tests/taskA/test_features_s10.py` | S0–S10, masks |
| `tests/taskA/test_decoder_kan.py` | KAN twin |
| `tests/taskA/test_config_hgt_fusion88.py` | Hydra freeze |

Only the 11.08 (HGT+S10) and 14.08 (MLP vs KAN) path remains.
