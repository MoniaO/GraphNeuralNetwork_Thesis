# Task A — Final Large Train Grid (11.08.2026)

Osobny blok eksperymentalny. **Nie miesza** strojenia architektury grafowej
z cechami statystycznymi.

## Zasada

| Stage | Cel | Statystyka |
|---|---|---|
| **A** | wybór backbone + hiperparametrów grafowych | `g_stat = 0` (24D zeros) |
| **B** | zamrożenie zwycięskiego backbone | brak zmian |
| **C** | porównanie wariantów S0–S10 | tylko tu zmienia się evidence |

**Test nie służy do selekcji** aż do zamrożenia architektury i reprezentacji.

## Protokół danych (wspólny)

- `candidate_seed = 20260722` (**nie** podąża za training seed)
- te same patient/candidate splits, `G_train`, node features, loss, AUPRC
- screening seeds: `20260721–23`
- final confirmation: `20260721–25`
- scenarios: wszystkie 6 GSN v3
- Stage A: `hcr=none`, `decoder=fusion88`

## Uruchomienie Stage A

```bash
export GSN_PROJECT_ROOT="$HOME/Desktop/GSN Graphs dysertation 2026"
export PYTHONPATH=src

# jednostkowe
.venv/bin/python -m pytest tests/test_fusion88_decoder_11_08_2026.py -q

# smoke (2 epoki, 1 konfiguracja HGT)
.venv/bin/python scripts/run_taskA_stage_a_backbone_11.08.2026.py --mode smoke

# pełny shared screen na clean × 3 seeds (~216 jobów)
.venv/bin/python scripts/run_taskA_stage_a_backbone_11.08.2026.py --mode shared_screen

# policz joby
.venv/bin/python scripts/run_taskA_stage_a_backbone_11.08.2026.py --mode count
```

## Struktura bloku

```
outputs/taskA_final_large_grid_11.08.2026/
  README_11.08.2026.md
  PROTOCOL_11.08.2026.md
  MANIFEST_11.08.2026.json
  stage_a/   stage_b/   stage_c/
  audit/     notebooks/ logs/

src/taskA_final_large_grid_11_08_2026/
configs/taskA_final_large_grid_11.08.2026/
scripts/run_taskA_stage_a_backbone_11.08.2026.py
notebooks/taskA_final_large_grid_11.08.2026/
tests/test_fusion88_decoder_11_08_2026.py
```

## Status

- [x] Blok + protokół + manifest
- [x] Fusion88 decoder (`4d→128→64`, g_stat zeros)
- [x] `rgcn_matched` + type-specific projection
- [x] Stage A grid + runner (candidate_seed frozen)
- [x] Smoke Stage A (HGT / GAT / RGCN, 2 epochs)
- [ ] Shared grid screen (~72 configs × 3 seeds = 216 jobs)
- [ ] Top-2 × 6 scenarios
- [ ] Stage B / C

Kod historyczny Wave 0–11 **nie jest nadpisywany**; ten blok jest addytywny.
