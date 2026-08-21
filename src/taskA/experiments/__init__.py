"""taskA.experiments — three stages in reproduction order.

  1. stage_a_backbone   encoder race (11.08) → freeze HGT
  2. stage_c_stats      S0–S10 on frozen HGT → freeze S10
  3. final_14_08        6 scenarios × 5 seeds × MLP/KAN

Outputs stay in:
  outputs/taskA_final_large_grid_11.08.2026/
  outputs/taskA_FINAL_14.08.2026/
"""
