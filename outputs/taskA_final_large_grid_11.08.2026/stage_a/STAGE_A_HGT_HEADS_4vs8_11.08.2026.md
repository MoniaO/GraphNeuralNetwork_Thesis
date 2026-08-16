# HGT heads refine — 4 vs 8 (11.08.2026)

Frozen base: `hgt h32 L2 d0.25 lr1e-3` on `clean` (no stats / g_stat=zeros).
heads=4 reused from Stage A shared freeze; heads=8 trained in refine/.

| heads | n_seeds | mean valid AUPRC ± std |
| ---: | ---: | ---: |
| 4 | 3 | 0.7288 ± 0.0091 |
| 8 | 3 | 0.7045 ± 0.0022 |

## Per-seed

- hd4 seed20260721: valid AUPRC=0.7223 (reused shared)
- hd4 seed20260722: valid AUPRC=0.7392 (reused shared)
- hd4 seed20260723: valid AUPRC=0.7249 (reused shared)
- hd8 seed20260721: valid AUPRC=0.7050
- hd8 seed20260722: valid AUPRC=0.7021
- hd8 seed20260723: valid AUPRC=0.7064

**Leader:** heads=4 (mean valid AUPRC 0.7288)

heads=8 refine **lost** (0.705 vs 0.729). Stage C started on **heads=4** (no backbone retune).
