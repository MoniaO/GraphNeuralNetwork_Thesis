# Kolejność przeglądu kodu — FINAL 14.08.2026 (dziś)

Cel: przejść **od decyzji → architektura → trening → diagnostyka**, bez gubienia kontekstu.
Czas orientacyjny: ~2–3 h uważnego czytania.

**Werdykt wyników (60/60):** MLP-stat wygrywa (macro valid **0.919** vs KAN **0.908**).  
Szczegóły: [`SUMMARY_MACRO_14.08.2026.md`](SUMMARY_MACRO_14.08.2026.md)

---

## 0. Najpierw kontekst (15 min)

1. [`00_README.md`](00_README.md) — co to jest FINAL  
2. [`PROTOCOL_14.08.2026.md`](PROTOCOL_14.08.2026.md) — freeze / zakazy  
3. [`MANIFEST_14.08.2026.json`](MANIFEST_14.08.2026.json) — fingerprint  
4. [`TESTS_SINCE_11.08.2026.md`](TESTS_SINCE_11.08.2026.md) — co już przetestowaliśmy od 11.08  
5. [`CODE_AND_SCRIPT_MAP_14.08.2026.md`](CODE_AND_SCRIPT_MAP_14.08.2026.md) — zależności skryptów (diagram)

---

## 1. Architektura modelu (rdzeń) — czytaj w tej kolejności

| # | Plik | Po co |
|---:|---|---|
| 1 | `src/taskA_final_large_grid_11_08_2026/fusion88_decoder.py` | kontrakt graph 64 + stat 24 → fusion 88 → logit |
| 2 | `src/taskA_final_large_grid_11_08_2026/stage_c/stat_encoder.py` | **MLP vs KAN** pair encoders, `g_stat`, reg loss |
| 3 | `src/taskA_final_large_grid_11_08_2026/stage_c/variants.py` | S0–S10, w tym S10 FULL40 = 40-D |
| 4 | `src/taskA_final_large_grid_11_08_2026/stage_c/features.py` | jak liczone są 40 cech (HCR/energy/marg/joint/META) |
| 5 | `src/taskA_final_large_grid_11_08_2026/stage_c/attach.py` | train-only fit, attach `stat_raw` + maski, leakage guards |
| 6 | `src/models/TaskA/hetero_gnn.py` | HGT + wiring `fusion88_stat` (szukaj `Fusion88Stat` / decoder) |
| 7 | `src/models/TaskA/pair_encoders/kan_linear.py` | implementacja KANLinear (używana przez StatKAN) |
| 8 | `src/train_taskA.py` | pętla train, early stop valid AUPRC, `pair_encoder_regularization_loss` |

---

## 2. FINAL runner (jak odpala się 60 jobów)

| # | Plik | Po co |
|---:|---|---|
| 1 | `src/taskA_FINAL_14_08_2026/__init__.py` | nazwa, seedy, freeze constants |
| 2 | `src/taskA_FINAL_14_08_2026/runner.py` | overrides Hydra, skip-ok, extract results |
| 3 | `scripts/run_taskA_FINAL_14.08.2026.py` | CLI (`count` / `smoke` / `full`) |
| 4 | `scripts/watch_FINAL_14.08.2026.py` | watchdog PID-file |

---

## 3. Kampania 11.08 (skąd wziął się freeze) — skrót

| # | Plik | Po co |
|---:|---|---|
| 1 | `src/taskA_final_large_grid_11_08_2026/__init__.py` | seedy / scenariusze |
| 2 | `src/taskA_final_large_grid_11_08_2026/stage_a_runner.py` | Stage A backbone race |
| 3 | `src/taskA_final_large_grid_11_08_2026/stage_c_runner.py` | Stage C S0–S10 + multi-scenario |
| 4 | `scripts/run_taskA_stage_a_backbone_11.08.2026.py` | CLI Stage A |
| 5 | `scripts/run_taskA_stage_c_stats_11.08.2026.py` | CLI Stage C |
| 6 | `outputs/.../stage_a/STAGE_A_FREEZE_HGT_TOP1_11.08.2026.md` | decyzja HGT |
| 7 | `outputs/.../stage_c/STAGE_C_CLEAN_S0_S10_TABLE_11.08.2026.md` | ranking S10/S9 |
| 8 | `outputs/.../stage_c/STAGE_C_S9_S10_6SCEN_TABLE_11.08.2026.md` | S9/S10 × 6 |

---

## 4. Diagnostyka (przejrzyj wyniki, potem kod)

English operator guide (all scripts + knobs):  
[`SCRIPTS_CONTROL_GUIDE_EN.md`](SCRIPTS_CONTROL_GUIDE_EN.md)

| # | Plik / folder | Po co |
|---:|---|---|
| 1 | `SUMMARY_MACRO_14.08.2026.md` | **tabela MLP vs KAN** |
| 2 | `FINAL_SUMMARY_14.08.2026.json` | raw per-seed |
| 3 | `learning_curves/LEARNING_DIAGNOSTICS_14.08.2026.md` | best-epoch / gap |
| 4 | `learning_curves/` PNG | train/valid curves |
| 5 | `scripts/plot_taska_learning_curves_14.08.2026.py` | jak parsowane logi |
| 6 | `scripts/eval_taska_stage_c_edge_pathway_report.py` | edge × prob × pathway |
| 7 | `edge_pathway_report/` | (uwaga: early report miał niski coverage join — do poprawy przy pushu jeśli potrzeba) |

---

## 5. Testy — uruchom / przeczytaj

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_fusion88_decoder_11_08_2026.py \
  tests/test_stage_c_stats_11_08_2026.py \
  tests/test_FINAL_14_08_2026_kan.py -q
```

| Plik | Co kryje |
|---|---|
| `tests/test_fusion88_decoder_11_08_2026.py` | fusion 88 shapes |
| `tests/test_stage_c_stats_11_08_2026.py` | S0–S10, features, masks |
| `tests/test_FINAL_14_08_2026_kan.py` | KAN twin shapes + reg |

---

## 6. Co **nie** mieszać przy review (osobny stack)

- `outputs/taskA_final_stack/` — Wave11 Wave7C / stary „final”
- `scripts/run_taska_mlp_vs_kan_scenarios.py` i Wave11 KAN — **inny** dekoder (nie fusion88 S10)

---

## Checklist „dziś done”

- [ ] Przeczytane §0–§2 (protokół + architektura + runner)
- [ ] Potwierdzony werdykt MLP > KAN w `SUMMARY_MACRO`
- [ ] Pytest zielony (3 pliki wyżej)
- [ ] Krzywe w `learning_curves/` wyglądają sensownie (brak eksplozji loss)
- [ ] Lista plików do **gita** uzgodniona (kod + md + summary JSON; bez ciężkich `.pt` / pełnych logów jeśli nie chcesz)

Jak skończysz review — napisz „push” / „commit”, wtedy dopiero zrobię commit+push.
