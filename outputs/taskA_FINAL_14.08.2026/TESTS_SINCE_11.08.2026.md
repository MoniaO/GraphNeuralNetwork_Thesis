# Testy od 11.08.2026 → FINAL 14.08.2026

Dokumentuje **jakie testy / ekrany** zrobiliśmy w kampanii od Stage A (11.08) do FINAL.

## A. Testy jednostkowe (pytest)

| Plik | Co sprawdza | Status |
|---|---|---|
| `tests/test_stage_c_stats_11_08_2026.py` | warianty S0–S10 dims; NMI/Jaccard/cosine; signed-phi bez silent binarize; Fusion88StatDecoder shapes; S0 zeros path | 8 passed (11.08) |
| `tests/test_FINAL_14_08_2026_kan.py` | StatKANPairEncoder 40→8; Fusion88StatDecoder `kan_shallow` forward; MLP reg=0; KAN reg≥0 | nowy (14.08) |
| `tests/test_fusion88_decoder_11_08_2026.py` | kontrakt fusion 88 (graph‖stat) | z kampanii 11.08 |

Uruchomienie:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_stage_c_stats_11_08_2026.py \
  tests/test_FINAL_14_08_2026_kan.py \
  tests/test_fusion88_decoder_11_08_2026.py -q
```

## B. Ekrany treningowe (eksperymenty) — 11.08

### Stage A — backbone race
- **288/288** jobów shared grid na `clean` × 3 seedy
- Backbones: `rgcn_matched`, `hgt`, `hetero_gatv2`, `hetero_sage_matched`
- Decoder Stage A: `fusion88` z `g_stat=zeros(24)`
- **Wynik:** HGT ≫ RGCN > SAGE > GAT
- **Freeze:** `hgt__h32__L2__d0.25__lr0.001__hd4` (valid ≈ 0.729)
- Heads refine: **hd4 0.729 > hd8 0.705** → Stage C tylko heads=4
- Artefakty: `outputs/taskA_final_large_grid_11.08.2026/stage_a/`

### Stage C — S0–S10 na clean
- **33/33** (11 wariantów × 3 seedy)
- Ranking: **S10 0.917** > **S9 0.913** > S1 0.853 > … > S0 0.719; META8/S8 szkodzą
- Tabela: `stage_c/STAGE_C_CLEAN_S0_S10_TABLE_11.08.2026.md`

### Stage C — S9+S10 × 6 scenariuszy
- **36/36** ok
- Macro valid: S9 **0.910** · S10 **0.915** (S10 wygrywa, zwłaszcza `selection_bias`)
- Tabela: `stage_c/STAGE_C_S9_S10_6SCEN_TABLE_11.08.2026.md`

### Protokół twardy (przez całą kampanię)
- Selection = **valid AUPRC** only
- `candidate_seed=20260722` **nie** śledzi training seed
- Test sealed (nie do wyboru)
- Stats fit train-only (bez G_true/labels leakage)
- Bez retune backbone + stats razem

## C. FINAL 14.08 — plan testów / jobów

| Etap | Zakres | Cel |
|---|---|---|
| Unit KAN | pytest wyżej | Kształty / reg loss |
| Learning curves | 18 runów S10 z 11.08 → PNG/CSV | Czy sieć się uczy, gap, best-epoch |
| Pathway early | eval na ckptach 11.08 S10 | Rare vs common `clinical_pathway` |
| Train FINAL | **60** jobów: S10 × {mlp, kan_shallow} × 6 × 5 | Ostateczny model + KAN twin |
| Pathway final | eval na ckptach FINAL | MLP vs KAN na rare ścieżkach |

## D. Co NIE jest testem FINAL

- Wave11 Wave7C / TA_MLP / TA_KAN_SHALLOW (`outputs/taskA_final_stack`, `wave11_taskA`) — **inny stack** (HGT L3/8heads + Wave7CDecoder). Nie mieszać liczb do tabel FINAL 14.08.
