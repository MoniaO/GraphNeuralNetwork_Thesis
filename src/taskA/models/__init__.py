"""taskA.models — graph encoder + Fusion88 edge decoder.

  encoder/           HGT (FINAL) and matched SAGE/GAT/RGCN (Stage A)
  decoder/           Fusion88 (64+24=88) and MLP/KAN pair encoder
  link_predictor.py  wires encoder to decoder

FINAL 14.08: encoder=hgt, decoder=fusion88_stat, pair encoder=mlp|kan_shallow.
"""
