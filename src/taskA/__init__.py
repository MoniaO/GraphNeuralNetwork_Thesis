"""Task A — directed-edge reconstruction on GSN v3.

Package layout (read top-down):

  taskA.data           graph, patients, and candidates
  taskA.features       S10 features (40D × AZ/AG/ZG), train-only attach
  taskA.models.encoder HGT (FINAL) plus SAGE/GAT/RGCN (Stage A)
  taskA.models.decoder Fusion88 + MLP/KAN pair encoder
  taskA.training       train loop / early stop / sealed test
  taskA.evaluation     AUPRC and auxiliary metrics
  taskA.experiments    Stage A → Stage C → FINAL 14.08

Frozen FINAL stack: HGT h32 L2 d0.25 heads=4 + fusion88_stat + S10.
Do not change hypers in `experiments.final_14_08` if you want the 14.08 table.
"""
