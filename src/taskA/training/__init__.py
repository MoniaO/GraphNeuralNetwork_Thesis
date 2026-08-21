"""taskA.training — one training loop for every stage.

train.py:
  1. loads the graph (taskA.data)
  2. if Stage C / FINAL: attach S10 (taskA.features.attach)
  3. builds LinkPredictor
  4. early-stops on valid AUPRC; test is computed once at the end

What you may change:
  training.epochs, patience, lr, seed, device, wandb.enabled
  (FINAL freezes these in experiments.final_14_08)

What not to touch for FINAL:
  selection_metric=auprc, sealed test, pos_weight from train
"""
