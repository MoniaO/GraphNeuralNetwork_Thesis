"""taskA.training — jedna pętla treningowa dla wszystkich etapów.

train.py:
  1. ładuje graf (taskA.data)
  2. jeśli Stage C / FINAL: attach S10 (taskA.features.attach)
  3. buduje LinkPredictor
  4. early stop na valid AUPRC; test liczony raz na końcu

Co wolno zmieniać:
  training.epochs, patience, lr, seed, device, wandb.enabled
  (FINAL zamraża te wartości w experiments.final_14_08)

Czego nie ruszać dla FINAL:
  selection_metric=auprc, test sealed, pos_weight z train
"""
