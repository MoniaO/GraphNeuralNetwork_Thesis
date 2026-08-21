# Skrypty Task A — kolejność odtwarzania

Wszystko leży w `scripts/taskA/`. Numer w nazwie = kolejność odpalania.

Ścieżka naukowa (to, co buduje wynik): **00 → 01 → 05 → 08 → 10 → 11**.
Watchdogi i podsumowania są opcjonalne.

| # | Plik | Kiedy | Co robi |
|---|---|---|---|
| 00 | `00_build_context_registry.py` | raz, przed S10 | rejestr koparentów Z → `outputs/wave5c/registry/` |
| 01 | `01_run_stage_a_backbone.py` | kampania 11.08 | wyścig encoderów (HGT wygrał) |
| 02 | `02_watch_stage_a.py` | opcjonalnie | watchdog Stage A |
| 03 | `03_summarize_stage_a.py` | po 01 | tabele decyzji Stage A |
| 04 | `04_upload_stage_a_curves.py` | opcjonalnie | krzywe Stage A → W&B |
| 05 | `05_run_stage_c_stats.py` | po freeze HGT | S0–S10 (S10 wygrał) |
| 06 | `06_watch_stage_c.py` | opcjonalnie | watchdog Stage C (clean) |
| 07 | `07_watch_stage_c_s9_s10.py` | opcjonalnie | watchdog S9/S10 × 6 scenariuszy |
| 08 | `08_run_final.py` | po freeze S10 | FINAL 14.08: 6 × 5 × MLP/KAN |
| 09 | `09_watch_final.py` | opcjonalnie | watchdog 60 jobów |
| 10 | `10_plot_learning_curves.py` | po 08 | CSV + PNG z logów |
| 11 | `11_eval_edge_pathway.py` | po 08 | predykcje × pathway / rare |

```bash
export GSN_PROJECT_ROOT="$HOME/Desktop/GSN Graphs dysertation 2026"
export PYTHONPATH="$PWD/src:$PWD"

# odtworzenie FINAL (zakłada gotowy rejestr 00 i freeze 11.08)
PYTHONPATH=src .venv/bin/python scripts/taskA/08_run_final.py --mode count
PYTHONPATH=src .venv/bin/python scripts/taskA/09_watch_final.py
PYTHONPATH=src .venv/bin/python scripts/taskA/10_plot_learning_curves.py --source final
PYTHONPATH=src .venv/bin/python scripts/taskA/11_eval_edge_pathway.py
```

Kod modelu jest w `src/taskA/`. Tutaj tylko entrypointy.
