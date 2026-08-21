"""taskA.experiments — trzy etapy w kolejności odtwarzania.

  1. stage_a_backbone   wyścig encoderów (11.08) → freeze HGT
  2. stage_c_stats      S0–S10 na zamrożonym HGT → freeze S10
  3. final_14_08        6 scenariuszy × 5 seedów × MLP/KAN

Outputy zostają w:
  outputs/taskA_final_large_grid_11.08.2026/
  outputs/taskA_FINAL_14.08.2026/
"""
