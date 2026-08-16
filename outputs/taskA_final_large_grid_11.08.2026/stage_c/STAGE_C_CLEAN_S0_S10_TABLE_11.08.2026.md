# Stage C clean screen — S0–S10 COMPLETE (11.08.2026)

**Frozen:** HGT `h32 L2 d0.25 lr1e-3` heads=4 · fusion88_stat · `candidate_seed=20260722`
**Selection:** mean valid AUPRC (3 seeds). Sealed test = info only.

| variant | mean valid AUPRC ± std | Δ vs S0 | mean sealed test AUPRC | mean valid AUC |
| --- | ---: | ---: | ---: | ---: |
| `S0_NONE` | 0.7186 ± 0.0147 | +0.0000 | 0.5569 | 0.8579 |
| `S1_NMI` | 0.8525 ± 0.0197 | +0.1339 | 0.7845 | 0.9233 |
| `S2_JACCARD_ACTIVE` | 0.7361 ± 0.0160 | +0.0175 | 0.6514 | 0.8974 |
| `S3_COSINE_ACTIVE` | 0.7959 ± 0.0239 | +0.0773 | 0.6376 | 0.9023 |
| `S4_HCR_BINARY_ONLY` | 0.8397 ± 0.0138 | +0.1211 | 0.8033 | 0.9250 |
| `S5_META8` | 0.6230 ± 0.0187 | -0.0956 | 0.4840 | 0.8314 |
| `S6_HCR_MATRIX16` | 0.8318 ± 0.0401 | +0.1132 | 0.8186 | 0.9320 |
| `S7_HCR_COMPACT24` | 0.7790 ± 0.0126 | +0.0604 | 0.7329 | 0.9161 |
| `S8_HCR_COMPACT32` | 0.6012 ± 0.0181 | -0.1175 | 0.4504 | 0.8307 |
| `S9_HCR_COMPACT36` | 0.9125 ± 0.0114 | +0.1939 | 0.9053 | 0.9578 |
| `S10_HCR_FULL40` | 0.9167 ± 0.0006 | +0.1981 | 0.9030 | 0.9571 |

## Ranking (Δ valid vs S0)

1. `S10_HCR_FULL40` — valid 0.9167 (Δ +0.1981), sealed test 0.9030
2. `S9_HCR_COMPACT36` — valid 0.9125 (Δ +0.1939), sealed test 0.9053
3. `S1_NMI` — valid 0.8525 (Δ +0.1339), sealed test 0.7845
4. `S4_HCR_BINARY_ONLY` — valid 0.8397 (Δ +0.1211), sealed test 0.8033
5. `S6_HCR_MATRIX16` — valid 0.8318 (Δ +0.1132), sealed test 0.8186
6. `S3_COSINE_ACTIVE` — valid 0.7959 (Δ +0.0773), sealed test 0.6376
7. `S7_HCR_COMPACT24` — valid 0.7790 (Δ +0.0604), sealed test 0.7329
8. `S2_JACCARD_ACTIVE` — valid 0.7361 (Δ +0.0175), sealed test 0.6514
9. `S0_NONE` — valid 0.7186 (Δ +0.0000), sealed test 0.5569
10. `S5_META8` — valid 0.6230 (Δ -0.0956), sealed test 0.4840
11. `S8_HCR_COMPACT32` — valid 0.6012 (Δ -0.1175), sealed test 0.4504

## Protocol P finalists (clean screen)

- `S0_NONE` — valid 0.7186 (Δ +0.0000), test 0.5569
- `S1_NMI` — valid 0.8525 (Δ +0.1339), test 0.7845
- `S9_HCR_COMPACT36` — valid 0.9125 (Δ +0.1939), test 0.9053
- `S10_HCR_FULL40` — valid 0.9167 (Δ +0.1981), test 0.9030

## Next

6 scenarios × 3 seeds on finalists (no retune). Then optional 5-seed confirm.

