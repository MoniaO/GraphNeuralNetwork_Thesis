"""Stage A — encoder race (11.08.2026).

What it does
------------
Compares SAGE / GATv2 / HGT / RGCN with g_stat=zeros(24).
Winner: HGT h32 L2 d0.25 lr1e-3 heads=4 (heads=8 lost the refine).

What you may change
-------------------
The grid in `grid.py` (hidden, layers, dropout, lr) — only if you rerun
the race. FINAL does not touch these constants.

Script: `scripts/taskA/01_run_stage_a_backbone.py`.
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
