# Graph Neural Network Thesis — Task A FINAL & Task B

Joint thesis repository on GSN v3:

- **Task A** — directed-edge reconstruction (heterogeneous link prediction);
- **Task B** — clinical endpoint prediction (multilabel node classification).

---

## Task A — link prediction (FINAL 14.08.2026)

The maintained path is **FINAL 14.08.2026**:

- encoder **HGT** `h32 · L2 · dropout 0.25 · lr 1e-3 · heads=4`;
- decoder **Fusion88-stat** + **S10_HCR_FULL40** features (40D × 3 roles AZ/AG/ZG);
- twin comparison **MLP-stat** vs **KAN-stat**;
- six patient scenarios × 5 seeds.

Verdict (60/60 jobs): **MLP-stat** is the final model
(macro valid AUPRC **0.919** vs KAN **0.908**).

```text
train patients ──► empirical node features ─┐
                                            ├─► HGT L2 ─► embeddings ─┐
positive train edges ─► G_train ────────────┘                         │
                                                                      ├─► Fusion88 → logit
S10 pair features AZ, AG, ZG ─► MLP or KAN ───────────────────────────┘
```

Patient rows are not GNN training examples. They supply node features and the
S10 fit, both restricted to the `train` partition.

## Architecture

Both twins share the encoder, candidate graph, splits, seeds, loss, and
evaluation protocol. They differ only in the S10 pair encoder:

- MLP-stat: `40 → 16 → 8`;
- KAN-stat: `StatKANPairEncoder` (shallow spline, same latent width).

Sources:

- `configs/model/hgt_fusion88.yaml` — frozen HGT + Fusion88-stat
- `configs/hcr/none.yaml` — classical HCR off; S10 enters through Stage C
- `src/taskA/models/` — HGT encoder and Fusion88 / MLP vs KAN decoder
- `src/taskA/experiments/` — Stage A → Stage C → FINAL 14.08

The 11.08 campaign selected this freeze (Stage A: HGT backbone; Stage C: S10).
Results: `outputs/taskA_final_large_grid_11.08.2026/` and
`outputs/taskA_FINAL_14.08.2026/`.

## Layout

```text
.
├── configs/                      Hydra: dataset_v3, hgt_fusion88, hcr=none
├── src/
│   ├── train_taskA.py            CLI (implementation: taskA.training.train)
│   ├── taskA/
│   │   ├── data/                 GSN v3 graph, patients, candidates
│   │   ├── features/             S0–S10, train-only attach, 40D bases
│   │   ├── models/encoder/       HGT (FINAL) + SAGE/GAT/RGCN
│   │   ├── models/decoder/       Fusion88 + MLP/KAN pair encoder
│   │   ├── training/             loop, early stop, sealed test
│   │   ├── evaluation/           AUPRC and diagnostics
│   │   └── experiments/          Stage A → Stage C → FINAL 14.08
│   └── taskB/                    endpoint prediction (see Task B section)
├── scripts/taskA/                00–11 in reproduction order
├── tests/taskA/                  Fusion88, S10, KAN, Hydra freeze
└── outputs/
    ├── taskA_FINAL_14.08.2026/
    └── taskA_final_large_grid_11.08.2026/
```

Details: `src/README.md`, `configs/README.md`, `scripts/taskA/README.md`,
`outputs/taskA_FINAL_14.08.2026/00_README.md`.

## Data

GSN v3 data lives outside this repository:

```bash
export GSN_PROJECT_ROOT="$HOME/Desktop/GSN Graphs dysertation 2026"
```

```text
$GSN_PROJECT_ROOT/
└── 2 v3. Data/
    ├── dataset_v3/
    │   ├── synthetic_pharmacotherapy_v3_nodes.csv
    │   ├── synthetic_pharmacotherapy_v3_edges_audited.csv
    │   └── synthetic_pharmacotherapy_v3_samples_<scenario>.csv
    └── splits/
        └── patient_splits_v3.csv
```

Scenarios: `clean`, `hidden_confounder`, `selection_bias`, `no_overlap`,
`noisy_documentation`, `multihospital`.

S10 attach needs the edge-context registry:

```text
outputs/taskA/context_registry/edge_context_registry.csv
```

If the file is missing:

```bash
PYTHONPATH=src .venv/bin/python scripts/taskA/00_build_context_registry.py
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PWD"
```

W&B is optional for tests. Full experiments use project `politechnika-gnn-thesis`.

## Quick check

```bash
.venv/bin/python -m pytest
```

Compose the config without training:

```bash
PYTHONPATH=src .venv/bin/python src/train_taskA.py \
  --cfg job \
  model=hgt_fusion88 \
  hcr=none \
  wandb.enabled=false
```

One-epoch smoke (no W&B):

```bash
PYTHONPATH=src .venv/bin/python src/train_taskA.py \
  model=hgt_fusion88 \
  hcr=none \
  data.dataset.scenario=clean \
  data.candidate_seed=20260722 \
  training.seed=20260722 \
  training.epochs=1 \
  training.device=cpu \
  wandb.enabled=false \
  ++experiment.stat_variant=S10_HCR_FULL40 \
  ++model.decoder.stat_pair_encoder=mlp
```

## FINAL 14.08 grid

```bash
PYTHONPATH=src .venv/bin/python scripts/taskA/08_run_final.py --mode count
PYTHONPATH=src .venv/bin/python scripts/taskA/09_watch_final.py
PYTHONPATH=src .venv/bin/python scripts/taskA/10_plot_learning_curves.py --source final
PYTHONPATH=src .venv/bin/python scripts/taskA/11_eval_edge_pathway.py
```

Heavy checkpoints and Hydra logs are gitignored. Versioned artefacts are the
light reports under `outputs/taskA_FINAL_14.08.2026/` and the 11.08 decision tables.

## Metrics

Model selection: validation AUPRC. Reports also include AUROC, Brier,
precision/recall/F1/F2, and clinical pathway metrics.

Loss: weighted `BCEWithLogitsLoss` (`pos_weight` from train). The classification
threshold is chosen on validation and frozen for test.

---

## Task B — endpoint prediction

Multilabel node classification: predict clinical endpoints
(`clinical_endpoint` nodes) from a per-patient causal graph.

Same audited DAG as Task A (`edges_audited.csv`). Unlike Task A, **each patient
is one training example**: one `HeteroData` with fixed topology and
patient-specific node values.

Current baseline:

- dataset **`dataset_syn`**, scenario **`clean`**, observability **`full`**
- encoder **HeteroConv** (`sage` / `transformer`) or **R-GCN**;
- 13 endpoints (AKI, DILI, Depression, …);
- checkpoint selection: validation **macro AUPRC**.

```text
audited DAG (nodes.csv + edges_audited.csv)     ← shared G_true, fixed topology
        │
per-patient values (samples_<scenario>.csv)   ← one graph instance per patient
        │  train-only standardisation (mean/std)
        │  observability mask (is_observed channel)
        │  structural leakage exclusions (latent nodes, endpoint descendants)
        ▼
HeteroData  →  GNN encoder  →  NodeClassificationHead  →  endpoint logits
```

**Task A vs Task B:**  Task B uses patients as **supervised
graph instances** with multilabel targets on endpoint nodes.

### Options (experiment knobs)

| Surface | Values / notes |
|---|---|
| **Backbone** | `TaskB_gnn_node` — `conv_type`: `sage`, `transformer` (HeteroConv) |
| | `TaskB_rgcn` — flattened node space + basis-decomposed R-GCN |
| **Dataset** | `dataset_syn` — GSN v3 synthetic pharmacotherapy (primary) |
| | `dataset_proxy` — Hetionet proxy benchmark |
| **Scenario** | `clean`, `hidden_confounder`, `selection_bias`, `no_overlap`, `noisy_documentation`, `multihospital` |
| **Observability regime** | `full` — almost all non-endpoint nodes visible |
| | `mechanisms_latent` — hide `mechanism` nodes (signal must propagate 2–5 hops) |
| | `bedside` — only `patient_context` visible at admission |
| **Targets** | default 10 endpoints in `configs/TaskB/data/dataset_syn.yaml`; override via `target=[...]` |
| **HCR fusion** (optional) | `model.use_hcr_wide`, `hcr_wide_mode`, `hcr_evidence_dim` (42D mixed evidence) |
| **Depth / regularisation** | `num_layers`, `use_residual`, `dropout`, `drop_edge`, `pair_norm_scale`, `jk_mode` |

Patient split: **70/15/15**, multilabel-balanced on `clean`, inherited by all
scenarios (`splits/patient_splits_v3.csv`). Normalisation stats: **train only**.

### Architecture (Task B)

Pipeline:

1. **Topology** — load shared hetero graph from audited edges; `edge_attr`
   includes effect size/sign and edge metadata.
2. **Per-patient graph** — node `x`: standardised realisation, `is_observed`
   flag, static node features, type identity embedding.
3. **Encoder** — stacked message passing with optional residual / JK / PairNorm.
4. **Class Model** — `TargetedPatientDAGNodeClassifier` (HeteroConv) or
   `RGCNPatientDAGNodeClassifier`; predicts a selected endpoint subset only.
5. **Training** — weighted `BCEWithLogitsLoss` (`pos_weight` per endpoint,
   capped); threshold chosen on validation when `auto_threshold: true`.

Sources:

- `configs/TaskB/config.yaml` — training loop, W&B, oversmoothing, early stopping
- `configs/TaskB/data/dataset_syn.yaml` — GSN v3 paths, scenario, regime, targets
- `configs/TaskB/data/dataset_proxy.yaml` — Hetionet proxy benchmark
- `configs/TaskB/model/TaskB_gnn_node.yaml` — HeteroConv backbone defaults
- `configs/TaskB/model/TaskB_rgcn.yaml` — R-GCN backbone defaults
- `src/taskB/data/PreprocessingTaskB/` — graph build, regimes, optional HCR features
- `src/taskB/models/TaskB/gnn_node.py` — HeteroConv classifier
- `src/taskB/models/TaskB/gnn_rgcn_node.py` — R-GCN classifier
- `src/taskB/models/TaskB/gnn_common.py` — shared head and label utilities
- `src/taskB/evaluation/syntetic_evaluator_node.py` — AUC/AUPRC and per-endpoint metrics
- `src/taskB/evaluation/oversmoothing_metrics.py` — Dirichlet energy, MAD, cosine, rank
- `src/taskB/train_taskB_opt.py` — Hydra CLI entrypoint

Details: `src/taskB/README.md`.

### Layout (Task B)

```text
.
├── configs/TaskB/
│   ├── config.yaml                 Hydra root (training, W&B, oversmoothing)
│   ├── data/
│   │   ├── dataset_syn.yaml        GSN v3 synthetic (primary)
│   │   └── dataset_proxy.yaml      Hetionet proxy benchmark
│   └── model/
│       ├── TaskB_gnn_node.yaml     sage/gcn/gat/transformer via HeteroConv
│       └── TaskB_rgcn.yaml         R-GCN path
└── src/taskB/
    ├── train_taskB_opt.py          CLI (Hydra + W&B)
    ├── data/PreprocessingTaskB/    per-patient HeteroData, regimes, HCR
    ├── models/TaskB/               encoders + classification head
    ├── evaluation/                 node metrics + oversmoothing
    └── training/                   losses, class weights
```

### Data (Task B)

Task B reads the **same v3 files** as Task A, but uses **patient rows as
labels**, not edge-candidate statistics.

Default training scenario: **`clean`**. Default observability: **`full`**.

Hetionet proxy (optional): set `data=dataset_proxy` and point
`data.dataset.root_dir` in `configs/TaskB/data/dataset_proxy.yaml` to your
local proxy export.

### Install (Task B)

Task B imports resolve from `src/taskB/`:

```bash
export PYTHONPATH="$PWD/src/taskB:$PWD/src"
```

### Metrics (Task B)

**Model selection:** validation **macro AP** (`training.selection_metric: auprc`).

| Metric | Notes |
|---|---|
| **AP** | macro over endpoints; primary selection metric |
| **AUROC** | macro over endpoints |
| **F1 / precision / recall** | macro; threshold from validation when `auto_threshold: true` |
| **Oversmoothing** | cosine, MAD, numerical rank, Dirichlet energy (optional, `track_oversmoothing: true`) |

**Loss:** weighted `BCEWithLogitsLoss`; `pos_weight` per endpoint from train
class balance, clipped by `training.pos_weight_cap` (default 30).

**W&B tags:** `conv_type`, observability `regime`, `scenario`, `layers`,
`residual`, `model`, `targets`.
