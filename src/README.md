# `src/` — Task A only

Single package: `taskA/`. CLI: `python src/train_taskA.py`.

Full map and reading order: [`taskA/README.md`](taskA/README.md).

```text
src/
  train_taskA.py     thin CLI (Hydra)
  taskA/
    data/            GSN v3 graph and candidates
    features/        S0–S10, attach, 40D bases, Z registry
    models/encoder/  HGT / SAGE / GAT / RGCN
    models/decoder/  Fusion88, MLP/KAN
    training/        training loop
    evaluation/      AUPRC and diagnostics
    experiments/     Stage A → Stage C → FINAL 14.08
```
