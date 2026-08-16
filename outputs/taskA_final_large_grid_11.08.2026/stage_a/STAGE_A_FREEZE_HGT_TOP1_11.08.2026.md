# Stage A freeze — HGT Top-1 (11.08.2026)

**Decision:** freeze backbone = **HGT Top-1**, user chose **L2**.

| field | value |
|---|---|
| config_id | `hgt__h32__L2__d0.25__lr0.001__hd4__shared` |
| backbone | `hgt` |
| hidden | 32 |
| layers | **2** |
| dropout | 0.25 |
| lr | 1e-3 |
| heads | 4 |
| decoder | `fusion88` (`g_stat=zeros`, no HCR) |
| candidate_seed | 20260722 |
| fingerprint | `0fe55c7a270f198b` |

**Clean screen (selection):** mean valid AUPRC **0.7288 ± 0.0091** (seeds 20260721/22/23).

Sealed test was **not** used for this choice.

## Heads refine — DONE

| heads | mean valid AUPRC |
| ---: | ---: |
| **4** | **0.729** (leader) |
| 8 | 0.705 (**lost**) |

→ keep **heads=4**. Do not retune backbone hypers for Stage C.

## Stage C

**Started** on frozen HGT (`h32 L2 d0.25 lr1e-3 heads=4`) — statistical variants S0–S10.
See `outputs/.../stage_c/` and `STAGE_C_STARTED_11.08.2026.md`.
