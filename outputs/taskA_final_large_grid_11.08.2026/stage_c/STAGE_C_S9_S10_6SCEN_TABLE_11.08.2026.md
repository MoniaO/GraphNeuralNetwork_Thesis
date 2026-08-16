# Stage C — S9 vs S10 on 6 scenarios (11.08.2026)

**Frozen:** HGT `h32 L2 d0.25 lr1e-3 heads=4` · `fusion88_stat` · `candidate_seed=20260722`
**Jobs:** 2 × 6 × 3 = **36/36 ok**, 0 fail. Selection = mean **valid AUPRC**; sealed test = info only.

| scenario | S9 valid ± std | S9 sealed test | S10 valid ± std | S10 sealed test | Δ valid (S10−S9) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `clean` | 0.913 ± 0.011 | 0.905 | 0.917 ± 0.001 | 0.903 | +0.004 |
| `hidden_confounder` | 0.912 ± 0.008 | 0.899 | 0.912 ± 0.013 | 0.902 | +0.001 |
| `selection_bias` | 0.884 ± 0.020 | 0.898 | 0.906 ± 0.014 | 0.903 | +0.023 |
| `no_overlap` | 0.910 ± 0.007 | 0.849 | 0.908 ± 0.033 | 0.887 | -0.002 |
| `noisy_documentation` | 0.911 ± 0.009 | 0.908 | 0.912 ± 0.014 | 0.902 | +0.001 |
| `multihospital` | 0.930 ± 0.007 | 0.862 | 0.932 ± 0.006 | 0.891 | +0.003 |
| **MACRO** | **0.910** | **0.887** | **0.915** | **0.898** | **+0.005** |

## Takeaways
- Both finalists hold ~0.91 valid across scenarios; **S10 wins macro** (0.915 vs 0.910).
- Largest S10 edge: **selection_bias** (+0.023 valid).
- S9 has larger val→test gaps on `no_overlap` and `multihospital` (info only; not for selection).

_Generated 2026-08-11T20:55Z_
