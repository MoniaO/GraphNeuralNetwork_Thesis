# Task A — mapa kodu

Rekonstrukcja skierowanych krawędzi na GSN v3. Czytaj katalogi **w tej kolejności**.

```text
src/taskA/
  data/            1. graf, pacjenci, kandydaci
  features/        2. cechy S10 (40D × AZ/AG/ZG), attach train-only
  models/encoder/  3. HGT (FINAL) + SAGE/GAT/RGCN (Stage A)
  models/decoder/  4. Fusion88 + MLP/KAN pair encoder
  models/          5. link_predictor.py skleja encoder z dekoderem
  training/        6. pętla train / early stop / test sealed
  evaluation/      7. AUPRC i metryki pomocnicze
  experiments/     8. Stage A → Stage C → FINAL 14.08
```

Entrypoint treningu: `src/train_taskA.py` → `taskA.training.train`.
Skrypty odtwarzania (numerowane): `scripts/taskA/00_…` … `11_…` — patrz `scripts/README.md`.

## Zamrożony stack FINAL 14.08

HGT `h32 · L2 · dropout 0.25 · lr 1e-3 · heads=4` + Fusion88-stat + **S10_HCR_FULL40**.
Twiny: MLP-stat vs KAN-stat. Selekcja: wyłącznie valid AUPRC.

**Nie zmieniaj** liczb z `experiments/final_14_08/__init__.py` ani wymiarów
`GRAPH_OUT=64`, `STAT_DIM=24`, `FUSION_DIM=88`, jeśli chcesz odtworzyć tabelę 14.08.

## Co wolno zmieniać (nowy eksperyment)

| Warstwa | Gdzie | Typowe gałki |
|---|---|---|
| dane | `configs/data/dataset_v3.yaml` | scenario, `candidate_seed` |
| encoder | `configs/model/hgt_fusion88.yaml` | hidden, layers, heads, dropout |
| dekoder | ten sam yaml, `decoder.*` | `stat_pair_encoder=mlp\|kan_shallow` |
| cechy | `++experiment.stat_variant=` | S0–S10 (FINAL = S10) |
| trening | `configs/config.yaml` → `training.*` | lr, epochs, patience, seed |

Każdy plik `.py` na powierzchni pakietu ma na górze: **co robi** i **czego nie ruszać dla FINAL**.
