"""Stage A — wyścig encoderów (11.08.2026).

Co robi
-------
Porównuje SAGE / GATv2 / HGT / RGCN przy g_stat=zeros(24).
Zwycięzca: HGT h32 L2 d0.25 lr1e-3 heads=4 (heads=8 przegrał refine).

Co wolno zmieniać
-----------------
Siatkę w `grid.py` (hidden, layers, dropout, lr) — tylko gdy powtarzasz
wyścig. FINAL nie rusza tych stałych.

Skrypt: `scripts/taskA/01_run_stage_a_backbone.py`.
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
