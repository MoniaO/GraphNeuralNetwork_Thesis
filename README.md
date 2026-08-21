# Graph Neural Network Thesis — Task A FINAL

Repozytorium pracy dyplomowej: rekonstrukcja skierowanych krawędzi grafu
przyczynowego farmakoterapii (heterogeneous link prediction na GSN v3).

Jedyna utrzymywana ścieżka to **FINAL 14.08.2026**:

- encoder **HGT** `h32 · L2 · dropout 0.25 · lr 1e-3 · heads=4`;
- dekoder **Fusion88-stat** + cechy **S10_HCR_FULL40** (40D × 3 role AZ/AG/ZG);
- porównanie twinów **MLP-stat** vs **KAN-stat**;
- sześć scenariuszy pacjentów × 5 seedów.

Werdykt (60/60 jobów): **MLP-stat** jest modelem finalnym
(macro valid AUPRC **0.919** vs KAN **0.908**).

```text
pacjenci train ──► empiryczne cechy węzłów ─┐
                                             ├─► HGT L2 ─► embeddingi ─┐
dodatnie krawędzie train ─► G_train ─────────┘                         │
                                                                       ├─► Fusion88 → logit
cechy S10 par AZ, AG, ZG ─► MLP albo KAN ──────────────────────────────┘
```

Wiersze pacjentów nie są przykładami treningowymi GNN. Służą do cech węzłów
i dopasowania S10 wyłącznie na partycji `train`.

## Architektura

Oba twiny mają identyczny encoder, graf kandydatów, splity, seedy, loss
i protokół ewaluacji. Różni je wyłącznie encoder pary S10:

- MLP-stat: `40 → 16 → 8`;
- KAN-stat: `StatKANPairEncoder` (płytki spline, ten sam wymiar).

Źródła:

- `configs/model/hgt_fusion88.yaml` — zamrożony HGT + Fusion88-stat
- `configs/hcr/none.yaml` — klasyczny HCR wyłączony; S10 wchodzi przez Stage C
- `src/taskA/models/` — encoder HGT i dekoder Fusion88 / MLP vs KAN
- `src/taskA/experiments/` — Stage A → Stage C → FINAL 14.08

Kampania 11.08 wybrała ten freeze (Stage A: backbone HGT; Stage C: S10).
Wyniki: `outputs/taskA_final_large_grid_11.08.2026/` oraz
`outputs/taskA_FINAL_14.08.2026/`.

## Struktura

```text
.
├── configs/                      Hydra: dataset_v3, hgt_fusion88, hcr=none
├── src/
│   ├── train_taskA.py            CLI (implementacja: taskA.training.train)
│   └── taskA/
│       ├── data/                 graf GSN v3, pacjenci, kandydaci
│       ├── features/             S0–S10, attach train-only, bazy 40D
│       ├── models/encoder/       HGT (FINAL) + SAGE/GAT/RGCN
│       ├── models/decoder/       Fusion88 + MLP/KAN pair encoder
│       ├── training/             pętla, early stop, test sealed
│       ├── evaluation/           AUPRC i diagnostyka
│       └── experiments/          Stage A → Stage C → FINAL 14.08
├── scripts/taskA/                00–11 w kolejności odtwarzania
├── tests/taskA/                  Fusion88, S10, KAN, Hydra freeze
└── outputs/
    ├── taskA_FINAL_14.08.2026/
    └── taskA_final_large_grid_11.08.2026/
```

Szczegóły: `src/README.md`, `configs/README.md`, `scripts/README.md`,
`outputs/taskA_FINAL_14.08.2026/00_README.md`.

## Dane

Dane GSN v3 są poza repozytorium:

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

Scenariusze: `clean`, `hidden_confounder`, `selection_bias`, `no_overlap`,
`noisy_documentation`, `multihospital`.

Attach S10 wymaga rejestru kontekstu krawędzi:

```text
outputs/taskA/context_registry/edge_context_registry.csv
```

Jeśli pliku nie ma:

```bash
PYTHONPATH=src .venv/bin/python scripts/taskA/00_build_context_registry.py
```

## Instalacja

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PWD"
```

W&B jest opcjonalne dla testów. Pełne eksperymenty używają projektu
`politechnika-gnn-thesis`.

## Szybka weryfikacja

```bash
.venv/bin/python -m pytest
```

Kompozycja konfiguracji bez treningu:

```bash
PYTHONPATH=src .venv/bin/python src/train_taskA.py \
  --cfg job \
  model=hgt_fusion88 \
  hcr=none \
  wandb.enabled=false
```

Jednoepokowy smoke (bez W&B):

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

## Finalny grid 14.08

```bash
PYTHONPATH=src .venv/bin/python scripts/taskA/08_run_final.py --mode count
PYTHONPATH=src .venv/bin/python scripts/taskA/09_watch_final.py
PYTHONPATH=src .venv/bin/python scripts/taskA/10_plot_learning_curves.py --source final
PYTHONPATH=src .venv/bin/python scripts/taskA/11_eval_edge_pathway.py
```

Ciężkie checkpointy i logi Hydra są w `.gitignore`. Wersjonowane są lekkie
raporty w `outputs/taskA_FINAL_14.08.2026/` oraz tabele decyzji 11.08.

## Metryki

Selekcja modelu: validation AUPRC. Raport zawiera też AUROC, Brier,
precision/recall/F1/F2 oraz metryki ścieżek klinicznych.

Loss: ważona `BCEWithLogitsLoss` (`pos_weight` z train). Próg klasyfikacji
wybierany na validation i zamrażany dla testu.
