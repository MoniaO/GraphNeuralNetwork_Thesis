# Task A — FINAL 14.08.2026

**Model name:** `FINAL_14.08.2026`  
**Stack:** HGT `h32 · L2 · d0.25 · lr1e-3 · heads=4` + `fusion88_stat` + **S10_HCR_FULL40**  
**Twins:** MLP-stat (default) and KAN-shallow-stat (fair twin)  
**Grid:** 6 scenarios × 5 `FINAL_SEEDS` × 2 encoders = **60/60 ok**  
**Selection:** valid AUPRC only · `candidate_seed=20260722` · test sealed

### Verdict

| | macro valid AUPRC | macro sealed test |
|---|---:|---:|
| **MLP-stat (FINAL)** | **0.919** | 0.896 |
| KAN-stat twin | 0.908 | 0.901 |
| Δ (KAN−MLP) | **−0.012** | +0.005 |

→ thesis model is **MLP-stat**. KAN is close but weaker on `selection_bias` (−0.017) and `multihospital` (−0.022).

This tree does **not** overwrite `outputs/taskA_final_large_grid_11.08.2026/`.

## Docs

1. **[`REVIEW_ORDER_TODAY.md`](REVIEW_ORDER_TODAY.md)** — code reading order
2. **[`SCRIPTS_CONTROL_GUIDE_EN.md`](SCRIPTS_CONTROL_GUIDE_EN.md)** — every script and what to control
3. [`SUMMARY_MACRO_14.08.2026.md`](SUMMARY_MACRO_14.08.2026.md) — result table
4. [`CODE_AND_SCRIPT_MAP_14.08.2026.md`](CODE_AND_SCRIPT_MAP_14.08.2026.md) — script dependencies
5. [`TESTS_SINCE_11.08.2026.md`](TESTS_SINCE_11.08.2026.md) — tests since 11.08
6. [`PROTOCOL_14.08.2026.md`](PROTOCOL_14.08.2026.md) / [`MANIFEST_14.08.2026.json`](MANIFEST_14.08.2026.json)

## Layout

```
outputs/taskA_FINAL_14.08.2026/
  REVIEW_ORDER_TODAY.md
  SUMMARY_MACRO_14.08.2026.md
  FINAL_SUMMARY_14.08.2026.json
  runs/{mlp|kan_shallow}/{scenario}/S10_HCR_FULL40/seed*/
  learning_curves/
  edge_pathway_report/
  logs/
```

## Commands

```bash
PYTHONPATH=src .venv/bin/python scripts/taskA/08_run_final.py --mode count
PYTHONPATH=src .venv/bin/python scripts/taskA/10_plot_learning_curves.py --source final
PYTHONPATH=src .venv/bin/python -m pytest tests/taskA -q
```
