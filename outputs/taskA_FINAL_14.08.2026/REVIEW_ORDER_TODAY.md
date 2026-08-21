# Kolejność przeglądu kodu — FINAL 14.08.2026

Cel: iść **dane → encoder → dekoder → trening → eksperymenty**, bez starych ścieżek.
Werdykt: MLP-stat (macro valid AUPRC **0.919**) vs KAN **0.908**.
Szczegóły: [`SUMMARY_MACRO_14.08.2026.md`](SUMMARY_MACRO_14.08.2026.md)

Konwencja nazw: `src/taskA/<etap>/` (data, features, models/encoder, models/decoder, …).
Skrypty: `scripts/taskA/00_…` … `11_…` w kolejności odtwarzania.

---

## 0. Kontekst (15 min)

1. [`00_README.md`](00_README.md) — co to jest FINAL
2. [`PROTOCOL_14.08.2026.md`](PROTOCOL_14.08.2026.md) — freeze / zakazy
3. [`src/taskA/README.md`](../../src/taskA/README.md) — mapa pakietu
4. [`scripts/README.md`](../../scripts/README.md) — numeracja 00–11

---

## 1. Architektura — czytaj w tej kolejności

| # | Plik | Po co |
|---:|---|---|
| 1 | `src/taskA/data/load_graph.py` | G_train + kandydaci |
| 2 | `src/taskA/features/variants.py` | S0–S10; FINAL = S10 40D |
| 3 | `src/taskA/features/attach.py` | train-only fit, `stat_raw` + maski |
| 4 | `src/taskA/models/encoder/hgt.py` | HGT L2 h32 |
| 5 | `src/taskA/models/decoder/fusion88.py` | graph 64 + stat 24 → 88 → logit |
| 6 | `src/taskA/models/decoder/pair_encoder.py` | **MLP vs KAN** |
| 7 | `src/taskA/models/link_predictor.py` | skleja encoder z dekoderem |
| 8 | `src/taskA/training/train.py` | early stop na valid AUPRC |

---

## 2. FINAL runner (60 jobów)

| # | Plik | Po co |
|---:|---|---|
| 1 | `src/taskA/experiments/final_14_08/__init__.py` | seedy, `FROZEN` |
| 2 | `src/taskA/experiments/final_14_08/runner.py` | Hydra overrides, skip-ok |
| 3 | `scripts/taskA/08_run_final.py` | CLI (`count` / `smoke` / `full`) |
| 4 | `scripts/taskA/09_watch_final.py` | watchdog |

---

## 3. Kampania 11.08 (skąd freeze)

| # | Plik | Po co |
|---:|---|---|
| 1 | `scripts/taskA/01_run_stage_a_backbone.py` | wyścig encoderów |
| 2 | `src/taskA/experiments/stage_a_backbone/` | siatka + runner Stage A |
| 3 | `scripts/taskA/05_run_stage_c_stats.py` | S0–S10 |
| 4 | `outputs/.../stage_a/STAGE_A_FREEZE_HGT_TOP1_11.08.2026.md` | decyzja HGT |
| 5 | `outputs/.../stage_c/STAGE_C_CLEAN_S0_S10_TABLE_11.08.2026.md` | ranking S10 |

---

## 4. Diagnostyka

| # | Plik | Po co |
|---:|---|---|
| 1 | `SUMMARY_MACRO_14.08.2026.md` | tabela MLP vs KAN |
| 2 | `scripts/taskA/10_plot_learning_curves.py` | krzywe z logów |
| 3 | `scripts/taskA/11_eval_edge_pathway.py` | krawędź × pathway |

---

## 5. Testy

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/taskA -q
```

| Plik | Co kryje |
|---|---|
| `tests/taskA/test_decoder_fusion88.py` | fusion 88 shapes |
| `tests/taskA/test_features_s10.py` | S0–S10, maski |
| `tests/taskA/test_decoder_kan.py` | KAN twin |
| `tests/taskA/test_config_hgt_fusion88.py` | Hydra freeze |

Stare fale Wave 0–11 i MOOC zostały usunięte. Jedyna ścieżka: 11.08 (HGT+S10) i 14.08 (MLP vs KAN).
