"""Task A FINAL model pack — 14.08.2026.

Frozen stack from Stage A/C (11.08.2026):
  HGT h32 L2 d0.25 lr1e-3 heads=4 + fusion88_stat + S10_HCR_FULL40

This pack re-trains fresh on all 6 scenarios × FINAL_SEEDS (5) for:
  - MLP-stat (default StatPairEncoder)
  - KAN-stat twin (StatKANPairEncoder)

Does NOT overwrite outputs/taskA_final_large_grid_11.08.2026/.
"""

BLOCK_DATE = "14.08.2026"
BLOCK_ID = "taskA_FINAL_14.08.2026"
MODEL_NAME = "FINAL_14.08.2026"
CANDIDATE_SEED = 20260722
FINAL_SEEDS = (20260721, 20260722, 20260723, 20260724, 20260725)
SCENARIOS = (
    "clean",
    "hidden_confounder",
    "selection_bias",
    "no_overlap",
    "noisy_documentation",
    "multihospital",
)
FROZEN = {
    "backbone": "hgt",
    "hidden_dim": 32,
    "n_layers": 2,
    "dropout": 0.25,
    "lr": 1e-3,
    "heads": 4,
    "weight_decay": 5e-4,
    "epochs": 200,
    "patience": 40,
    "grad_clip": 1.0,
    "stat_variant": "S10_HCR_FULL40",
    "decoder": "fusion88_stat",
}
