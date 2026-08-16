# Task A — FINAL 14.08.2026

**Model name:** `FINAL_14.08.2026`  
**Stack:** HGT `h32 · L2 · d0.25 · lr1e-3 · heads=4` + `fusion88_stat` + **S10_HCR_FULL40**  
**Twins:** MLP-stat (default) and KAN-shallow-stat (fair twin)  
**Grid:** 6 scenarios × 5 `FINAL_SEEDS` × 2 encoders = **60/60 ok**  
**Selection:** valid AUPRC only · `candidate_seed=20260722` · test sealed

### Werdykt

| | macro valid AUPRC | macro sealed test |
|---|---:|---:|
| **MLP-stat (FINAL)** | **0.919** | 0.896 |
| KAN-stat twin | 0.908 | 0.901 |
| Δ (KAN−MLP) | **−0.012** | +0.005 |

→ do tezy bierzemy **MLP-stat**. KAN blisko, ale słabszy zwłaszcza na `selection_bias` (−0.017) i `multihospital` (−0.022).

This tree does **not** overwrite `outputs/taskA_final_large_grid_11.08.2026/`.

## Docs (przegląd dziś)

1. **[`REVIEW_ORDER_TODAY.md`](REVIEW_ORDER_TODAY.md)** ← kolejność czytania kodu
2. **[`SCRIPTS_CONTROL_GUIDE_EN.md`](SCRIPTS_CONTROL_GUIDE_EN.md)** ← **English:** every script, what it does, what to control (for FINAL push)
3. [`SUMMARY_MACRO_14.08.2026.md`](SUMMARY_MACRO_14.08.2026.md) — tabela wyników
4. [`CODE_AND_SCRIPT_MAP_14.08.2026.md`](CODE_AND_SCRIPT_MAP_14.08.2026.md) — zależności skryptów
5. [`TESTS_SINCE_11.08.2026.md`](TESTS_SINCE_11.08.2026.md) — testy od 11.08
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
PYTHONPATH=src .venv/bin/python scripts/run_taskA_FINAL_14.08.2026.py --mode count
PYTHONPATH=src .venv/bin/python scripts/plot_taska_learning_curves_14.08.2026.py --source both
PYTHONPATH=src .venv/bin/python -m pytest tests/test_FINAL_14_08_2026_kan.py tests/test_stage_c_stats_11_08_2026.py -q
```
