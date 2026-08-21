# PROTOCOL — FINAL 14.08.2026

## Freeze (do not change)

| Field | Value |
|---|---|
| backbone | HGT |
| hidden | 32 |
| layers | 2 |
| dropout | 0.25 |
| lr | 1e-3 |
| heads | 4 |
| weight_decay | 5e-4 |
| epochs / patience | 200 / 40 |
| decoder | `fusion88_stat` |
| stat variant | `S10_HCR_FULL40` (D=40) |
| candidate_seed | **20260722** (fixed) |
| training seeds | 20260721 … 20260725 (5) |
| selection | valid AUPRC |
| test | sealed |

## Twins

- **MLP:** `++model.decoder.stat_pair_encoder=mlp` (D→16→8)
- **KAN:** `++model.decoder.stat_pair_encoder=kan_shallow` (KANLinear D→8 + LN); `spline_l1=1e-5`

## Do not

- Retune HGT hypers or S10 slicing
- Use test AUPRC for model selection
- Overwrite `outputs/taskA_final_large_grid_11.08.2026/`
- Mix numbers from older stacks into the FINAL 14.08 table
