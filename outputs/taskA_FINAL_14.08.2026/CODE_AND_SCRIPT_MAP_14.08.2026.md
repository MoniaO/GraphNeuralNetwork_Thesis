# FINAL 14.08.2026 — code and script map

Convention: `src/taskA/<stage>/` and `scripts/taskA/00_…11_`.

## Diagram

```mermaid
flowchart TB
  subgraph train_core [Training]
    TT[src/taskA/training/train.py]
    LD[taskA.data.load_graph]
    ATT[taskA.features.attach]
    FEAT[taskA.features.compute S10]
    DEC[taskA.models.decoder.pair_encoder]
    HGT[taskA.models.encoder.hgt]
    TT --> LD --> ATT --> FEAT
    TT --> HGT --> DEC
  end

  subgraph stage_1108 [11.08 campaign]
    SA[scripts/taskA/01_run_stage_a_backbone.py]
    SC[scripts/taskA/05_run_stage_c_stats.py]
    SA --> TT
    SC --> TT
  end

  subgraph final_1408 [FINAL 14.08]
    FR[scripts/taskA/08_run_final.py]
    RUN[taskA.experiments.final_14_08.runner]
    WD[scripts/taskA/09_watch_final.py]
    FR --> RUN --> TT
    WD --> FR
  end

  subgraph diag [Diagnostics]
    CUR[scripts/taskA/10_plot_learning_curves.py]
    PATH[scripts/taskA/11_eval_edge_pathway.py]
    CUR --> LOGS[train logs]
    PATH --> CKPT[best_model.pt]
  end
```

Reproduction order: **00 → 01 → 05 → 08 → 10 → 11** (02, 03, 04, 06, 07, 09 optional).

## FINAL 14.08

| Path | Role |
|---|---|
| `src/taskA/experiments/final_14_08/` | seeds, freeze, 60-job runner |
| `scripts/taskA/08_run_final.py` | CLI |
| `scripts/taskA/09_watch_final.py` | watchdog |
| `scripts/taskA/10_plot_learning_curves.py` | curves |
| `scripts/taskA/11_eval_edge_pathway.py` | pathway report |
| `src/taskA/models/decoder/pair_encoder.py` | MLP vs KAN |

## 11.08 campaign (do not overwrite outputs)

| Path | Role |
|---|---|
| `src/taskA/experiments/stage_a_backbone/` | encoder race |
| `src/taskA/experiments/stage_c_stats/` | S0–S10 |
| `src/taskA/models/decoder/fusion88.py` | 64+24=88 contract |
| `scripts/taskA/00_build_context_registry.py` | Z registry |
| `outputs/taskA_final_large_grid_11.08.2026/` | freeze + tables |

## One FINAL job

1. `runner.build_overrides` → HGT freeze + `fusion88_stat` + S10 + mlp/kan
2. `train.main` loads GSN v3
3. `fit_and_attach_stage_c_stats` on **train patients** → `stat_raw [N,3,40]`
4. HGT L=2 → 32-D embeddings
5. Decoder: `g_graph∈R64` + `g_stat∈R24` → fusion 88 → logit
6. Early stop on valid AUPRC; test sealed
