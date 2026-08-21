# Tests from 11.08.2026 → FINAL 14.08.2026

What screens ran from Stage A (11.08) through FINAL.

## A. Unit tests (pytest)

| File | Checks | Status |
|---|---|---|
| `tests/taskA/test_features_s10.py` | S0–S10 dims; NMI/Jaccard/cosine; signed-phi without silent binarize; Fusion88StatDecoder shapes; S0 zeros path | passed |
| `tests/taskA/test_decoder_kan.py` | StatKANPairEncoder 40→8; Fusion88StatDecoder `kan_shallow` forward; MLP reg=0; KAN reg≥0 | passed |
| `tests/taskA/test_decoder_fusion88.py` | fusion 88 contract (graph‖stat) | passed |
| `tests/taskA/test_config_hgt_fusion88.py` | Hydra freeze `model=hgt_fusion88` | passed |

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/taskA -q
```

## B. Training screens — 11.08

### Stage A — backbone race
- **288/288** shared-grid jobs on `clean` × 3 seeds
- Backbones: `rgcn_matched`, `hgt`, `hetero_gatv2`, `hetero_sage_matched`
- Stage A decoder: `fusion88` with `g_stat=zeros(24)`
- **Result:** HGT ≫ RGCN > SAGE > GAT
- **Freeze:** `hgt__h32__L2__d0.25__lr0.001__hd4` (valid ≈ 0.729)
- Heads refine: **hd4 0.729 > hd8 0.705** → Stage C uses heads=4 only
- Artefacts: `outputs/taskA_final_large_grid_11.08.2026/stage_a/`

### Stage C — S0–S10 on clean
- **33/33** (11 variants × 3 seeds)
- Ranking: **S10 0.917** > **S9 0.913** > S1 0.853 > … > S0 0.719; META8/S8 hurt
- Table: `stage_c/STAGE_C_CLEAN_S0_S10_TABLE_11.08.2026.md`

### Stage C — S9+S10 × 6 scenarios
- **36/36** ok
- Macro valid: S9 **0.910** · S10 **0.915** (S10 wins, especially `selection_bias`)
- Table: `stage_c/STAGE_C_S9_S10_6SCEN_TABLE_11.08.2026.md`

### Hard protocol (whole campaign)
- Selection = **valid AUPRC** only
- `candidate_seed=20260722` does **not** follow the training seed
- Test sealed (not used for choice)
- Stats fit train-only (no G_true/label leakage)
- No joint retune of backbone + stats

## C. FINAL 14.08 — jobs

| Stage | Scope | Goal |
|---|---|---|
| Unit KAN | pytest above | shapes / reg loss |
| Learning curves | 18 S10 runs from 11.08 → PNG/CSV | learning, gap, best-epoch |
| Pathway early | eval on 11.08 S10 ckpts | rare vs common `clinical_pathway` |
| Train FINAL | **60** jobs: S10 × {mlp, kan_shallow} × 6 × 5 | final model + KAN twin |
| Pathway final | eval on FINAL ckpts | MLP vs KAN on rare paths |

## D. Not a FINAL test

Older stacks are a different architecture. Do not mix those numbers into the FINAL 14.08 table.
