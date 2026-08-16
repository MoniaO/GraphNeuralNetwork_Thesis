"""Task A Final Large Train Grid — block 11.08.2026.

Stage A: backbone race with g_stat=zeros(24).
Stage B: freeze winner (HGT h32 L2 d0.25 lr1e-3 heads=4; heads=8 lost refine).
Stage C: statistical variants S0–S10 on frozen backbone (`stage_c/`).
"""

BLOCK_DATE = "11.08.2026"
BLOCK_ID = "taskA_final_large_grid_11.08.2026"
CANDIDATE_SEED = 20260722
SCREENING_SEEDS = (20260721, 20260722, 20260723)
FINAL_SEEDS = (20260721, 20260722, 20260723, 20260724, 20260725)
SCENARIOS = (
    "clean",
    "hidden_confounder",
    "selection_bias",
    "no_overlap",
    "noisy_documentation",
    "multihospital",
)
