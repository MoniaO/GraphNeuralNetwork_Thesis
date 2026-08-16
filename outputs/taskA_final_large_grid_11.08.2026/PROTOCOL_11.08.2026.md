# PROTOCOL — Task A Final Large Grid (11.08.2026)

Source instruction frozen into this block. Do not retune mid-flight.

## Hard stops

STOP if any of:

- candidate fingerprint changes between runs
- `candidate_seed` changes with `training_seed`
- patient split / `G_train` changes
- test used for model selection
- `G_true` or edge labels enter MI/Jaccard/Cosine/HCR fitting
- val/test patients enter statistical fitting
- continuous/count silently binarized for `HCR_BINARY_ONLY`
- GAT drops relation types
- statistical variants use different final decoders
- hyperparameters retuned per statistical feature
- FULL40 formulas changed during the experiment

## Stage A — backbone race (no statistical evidence)

Decoder (identical for all backbones) — `Fusion88Decoder`:

```
q_AG = [z_A || z_G || z_A⊙z_G || |z_A−z_G|]   # 4·d
→ Linear(4d, 128) → GELU → Dropout(0.2)
→ Linear(128, 64) → GELU → LayerNorm → g_graph ∈ R^64

g_stat = zeros(24)          # Stage A only

fusion [g_graph || g_stat] ∈ R^88
→ Linear(88,64) → GELU → Dropout(0.2) → Linear(64,1)  logit
```

File: `src/taskA_final_large_grid_11_08_2026/fusion88_decoder.py`

Loss: `BCEWithLogitsLoss` with `pos_weight = n_neg/n_pos`.

Shared grid:

| hyper | values |
|---|---|
| hidden_dim | 32, 64 |
| n_layers | 2, 3, 4 |
| dropout | 0.10, 0.25 |
| lr | 1e-3, 3e-4 |

Fixed: `weight_decay=5e-4`, `epochs=200`, `patience=40`, `grad_clip=1.0`.

Backbones: `rgcn_matched`, `hgt`, `hetero_gatv2` (relation-aware; no collapse).

After shared grid — architecture-specific refinement only:

- HGT heads ∈ {4,8}
- GAT heads ∈ {2,4,8}
- RGCN num_bases ∈ {4,8,full}

Selection path:

1. Screen on **clean**, 3 seeds → Top-2 configs / backbone
2. Top-2 × 6 scenarios × 3 seeds
3. Freeze one config / backbone
4. Final backbone comparison: 6 × 5 seeds  
   Primary: **macro validation AUPRC** across scenarios  
   Secondary: Brier, AUROC, seed SD  
   **No test peek**

## Stage B — freeze winner

No changes to backbone / dims / layers / heads|bases / dropout / LR /
edge decoder / training protocol.

## Stage C — statistical variants S0–S10

Only on frozen winner. Motif roles AZ, AG, ZG; missing Z → masks `[0,1,0]`.

See instruction sections H–P for MI / Jaccard_ACTIVE / Cosine_ACTIVE /
HCR_BINARY_ONLY / mixed HCR / META8 / HCR16…FULL40.

Common pair encoder: `D→16→GELU→LN→Drop0.1→8→LN`, unshared roles → `g_stat∈R^24`.

## Scientific questions (must answer after Stage C)

1–11 as in the instruction (backbone without stats; MI/J/C sufficiency;
BB-only HCR; mixed HCR on count/continuous; META8 shortcut; HCR16 above META8;
marginals; joint; energy redundancy; smallest adequate HCR).
