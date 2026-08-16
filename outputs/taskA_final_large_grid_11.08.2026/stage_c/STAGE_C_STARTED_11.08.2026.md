# Stage C started — 11.08.2026

**Frozen backbone (no retune):** HGT `h32 L2 d0.25 lr1e-3` **heads=4**

- Stage A clean mean valid AUPRC ≈ 0.729
- heads=8 refine **lost** (≈ 0.705) — not used for Stage C

**Decoder:** Fusion88 (+ StatPairEncoder D→16→8 unshared AZ/AG/ZG → `g_stat`∈R^24).  
S0 uses exact zeros path (`Fusion88Decoder` / `force_zero_stat`).

**Protocol:** `hcr=none` (Stage C owns stats); `candidate_seed=20260722`; train seeds 20260721/22/23; selection = **valid AUPRC** only.

**Screen (default):** `clean` × S0–S10 × 3 seeds.

```bash
PYTHONPATH=src GSN_PROJECT_ROOT="$HOME/Desktop/GSN Graphs dysertation 2026" \
  .venv/bin/python scripts/run_taskA_stage_c_stats_11.08.2026.py \
  --mode screen --scenario clean --heads 4
```
