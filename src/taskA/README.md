# Task A — code map

Directed-edge reconstruction on GSN v3. Read the packages **in this order**.

```text
src/taskA/
  data/            1. graph, patients, candidates
  features/        2. S10 features (40D × AZ/AG/ZG), train-only attach
  models/encoder/  3. HGT (FINAL) + SAGE/GAT/RGCN (Stage A)
  models/decoder/  4. Fusion88 + MLP/KAN pair encoder
  models/          5. link_predictor.py wires encoder to decoder
  training/        6. train loop / early stop / sealed test
  evaluation/      7. AUPRC and auxiliary metrics
  experiments/     8. Stage A → Stage C → FINAL 14.08
```

Training entrypoint: `src/train_taskA.py` → `taskA.training.train`.
Reproduction scripts (numbered): `scripts/taskA/00_…` … `11_…` — see
`scripts/taskA/README.md`.

## Frozen FINAL 14.08 stack

HGT `h32 · L2 · dropout 0.25 · lr 1e-3 · heads=4` + Fusion88-stat + **S10_HCR_FULL40**.
Twins: MLP-stat vs KAN-stat. Selection: valid AUPRC only.

**Do not change** the numbers in `experiments/final_14_08/__init__.py` or the
dims `GRAPH_OUT=64`, `STAT_DIM=24`, `FUSION_DIM=88` if you want to reproduce
the 14.08 table.

## What you may change (new experiment)

| Layer | Where | Typical knobs |
|---|---|---|
| data | `configs/data/dataset_v3.yaml` | scenario, `candidate_seed` |
| encoder | `configs/model/hgt_fusion88.yaml` | hidden, layers, heads, dropout |
| decoder | same yaml, `decoder.*` | `stat_pair_encoder=mlp\|kan_shallow` |
| features | `++experiment.stat_variant=` | S0–S10 (FINAL = S10) |
| training | `configs/config.yaml` → `training.*` | lr, epochs, patience, seed |

Each surface `.py` file starts with: **what it does** and **what not to touch for FINAL**.
