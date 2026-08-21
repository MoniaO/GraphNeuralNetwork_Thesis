# Konfiguracje Task A

Hydra składa dane, model i flagę HCR. Publiczny model: **`hgt_fusion88`**.

## Grupy

| Plik | Rola | Co wolno zmieniać |
|---|---|---|
| `config.yaml` | trening, W&B, experiment | epochs, patience, lr, seed, `wandb.enabled` |
| `data/dataset_v3.yaml` | GSN v3 + scenariusz | `scenario`, `candidate_seed` (FINAL = 20260722) |
| `model/hgt_fusion88.yaml` | freeze HGT + Fusion88-stat | hypers encodera/dekodera — **nie ruszać dla 14.08** |
| `hcr/none.yaml` | klasyczny HCR wyłączony | zostaw `enabled: false`; S10 wchodzi przez Stage C |
| `taskA/experiments/stage_a_backbone.yaml` | protokół Stage A | tylko gdy powtarzasz wyścig backbone |
| `taskA/features/context_registry.yaml` | budowa rejestru Z | ścieżki i seedy rejestru |

Stage A / Stage C / FINAL nadpisują pola modelu w runnerze.

```bash
PYTHONPATH=src .venv/bin/python src/train_taskA.py \
  --cfg job \
  model=hgt_fusion88 \
  hcr=none \
  wandb.enabled=false
```
