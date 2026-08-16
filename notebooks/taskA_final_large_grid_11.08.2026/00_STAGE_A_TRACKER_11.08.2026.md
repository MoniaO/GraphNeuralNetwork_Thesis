# Stage A tracker — 11.08.2026

Use this notebook folder for analysis notebooks. Prefer filenames:

`NN_description_11.08.2026.ipynb`

## Checklist

1. Smoke (`--mode smoke`) passes with `fusion88` + `hcr=none`
2. Shared grid screen on `clean` × 3 seeds
3. **STOP** — decision table → pick Top-2 / backbone (before any stats layer) ✅
4. **FROZEN (user):** HGT Top-1 = `hgt__h32__L2__d0.25__lr0.001__hd4__shared`  
   → `outputs/.../stage_a/STAGE_A_FREEZE_HGT_TOP1_11.08.2026.{md,json}`
5. Heads refine 4 vs 8 on frozen HGT — **DONE** (hd4=0.729 wins; hd8=0.705 lost)
6. Final backbone comparison 6 × 5 seeds (valid macro AUPRC only) — if still needed
7. **Do not** open sealed test metrics for selection
8. Stage C stats — **STARTED** on heads=4 (`scripts/run_taskA_stage_c_stats_11.08.2026.py`)

## Decision artifacts (auto)

- `outputs/taskA_final_large_grid_11.08.2026/stage_a/STAGE_A_BACKBONE_DECISION_TABLE_11.08.2026.md`
- `.../STAGE_A_BACKBONE_RANKING_11.08.2026.csv`
- Refresh: `python scripts/summarize_stage_a_backbone_11.08.2026.py --scenario clean`

## Backbones in shared screen

`rgcn_matched` · `hgt` · `hetero_gatv2` · `hetero_sage_matched` (GraphSAGE)

Planned jobs: **96 configs × 3 seeds = 288** (SAGE added mid-run; same grid, no stats).
