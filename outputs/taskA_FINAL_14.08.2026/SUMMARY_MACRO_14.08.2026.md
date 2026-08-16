# FINAL 14.08.2026 — SUMMARY MLP vs KAN

_Generated 2026-08-15T09:28Z_

Progress: MLP **30/30** · KAN **30/30** · total **60/60**

| scenario | MLP valid ± std | MLP sealed test | KAN valid ± std | KAN sealed test | Δ valid (KAN−MLP) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `clean` | 0.919 ± 0.010 (n=5) | 0.904 | 0.911 ± 0.015 (n=5) | 0.913 | -0.009 |
| `hidden_confounder` | 0.918 ± 0.007 (n=5) | 0.902 | 0.912 ± 0.013 (n=5) | 0.906 | -0.006 |
| `selection_bias` | 0.909 ± 0.008 (n=5) | 0.903 | 0.892 ± 0.014 (n=5) | 0.903 | -0.017 |
| `no_overlap` | 0.911 ± 0.021 (n=5) | 0.881 | 0.902 ± 0.012 (n=5) | 0.884 | -0.008 |
| `noisy_documentation` | 0.917 ± 0.006 (n=5) | 0.903 | 0.908 ± 0.010 (n=5) | 0.914 | -0.009 |
| `multihospital` | 0.943 ± 0.009 (n=5) | 0.880 | 0.920 ± 0.013 (n=5) | 0.883 | -0.022 |
| **MACRO** | **0.919** | **0.896** | **0.908** | **0.901** | **-0.012** |

## Verdict

- **Winner FINAL:** **MLP-stat** (macro valid MLP 0.919 vs KAN 0.908).
- Selection = valid AUPRC; test sealed.
- Stack: HGT L2 h32 hd4 + fusion88_stat + S10_HCR_FULL40.
