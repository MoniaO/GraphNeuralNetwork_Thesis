# Task A configs

Hydra composes data, model, and the HCR flag. Public model: **`hgt_fusion88`**.

## Groups

| File | Role | What you may change |
|---|---|---|
| `config.yaml` | training, W&B, experiment | epochs, patience, lr, seed, `wandb.enabled` |
| `data/dataset_v3.yaml` | GSN v3 + scenario | `scenario`, `candidate_seed` (FINAL = 20260722) |
| `model/hgt_fusion88.yaml` | freeze HGT + Fusion88-stat | encoder/decoder hypers — **do not touch for 14.08** |
| `hcr/none.yaml` | classical HCR off | leave `enabled: false`; S10 enters through Stage C |
| `taskA/experiments/stage_a_backbone.yaml` | Stage A protocol | only if you rerun the backbone race |
| `taskA/features/context_registry.yaml` | Z registry build | registry paths and seeds |

Stage A / Stage C / FINAL override model fields in the runner.

```bash
PYTHONPATH=src .venv/bin/python src/train_taskA.py \
  --cfg job \
  model=hgt_fusion88 \
  hcr=none \
  wandb.enabled=false
```
