# Graph Neural Network Thesis — Task A FINAL

Thesis repository: directed-edge reconstruction on a pharmacotherapy causal graph
(heterogeneous link prediction on GSN v3).

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
│   └── taskA/
│       ├── data/                 GSN v3 graph, patients, candidates
│       ├── features/             S0–S10, train-only attach, 40D bases
│       ├── models/encoder/       HGT (FINAL) + SAGE/GAT/RGCN
│       ├── models/decoder/       Fusion88 + MLP/KAN pair encoder
│       ├── training/             loop, early stop, sealed test
│       ├── evaluation/           AUPRC and diagnostics
│       └── experiments/          Stage A → Stage C → FINAL 14.08
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
