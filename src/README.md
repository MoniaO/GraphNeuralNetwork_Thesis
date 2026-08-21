# `src/` — tylko Task A

Jedyny pakiet: `taskA/`. CLI: `python src/train_taskA.py`.

Pełna mapa i kolejność czytania: [`taskA/README.md`](taskA/README.md).

```text
src/
  train_taskA.py     cienki CLI (Hydra)
  taskA/
    data/            graf GSN v3 i kandydaci
    features/        S0–S10, attach, bazy 40D, rejestr Z
    models/encoder/  HGT / SAGE / GAT / RGCN
    models/decoder/  Fusion88, MLP/KAN
    training/        pętla treningowa
    evaluation/      AUPRC i diagnostyka
    experiments/     Stage A → Stage C → FINAL 14.08
```
