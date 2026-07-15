# GraphNeuralNetwork_Thesis
This thesis investigates the oversmoothing phenomenon in Graph Neural Networks 
(GNNs) applied to heterogeneous graphs with rare causal edges. We generate a 
synthetic heterogeneous graph with a controlled causal structure, enabling 
precise evaluation of model behavior under known ground truth. Using this graph, 
we benchmark selected GNN architectures on a link prediction task, with emphasis 
on three research dimensions: 
1. model robustness to oversmoothing as network depth increases, 
2. ability to detect rare causal relations under class imbalance, 
3. prediction stability under distribution shift caused by spurious correlations and varying data-generating environments.

## Project Overview

Experiments are conducted on two datasets:
1. **`ogbl-ddi`** — drug-drug interaction network (Open Graph Benchmark)
2. **Synthetic dataset** — medical data with rare edges and casuality

---

## Repository Structure
```
GraphNeuralNetwork_Thesis/
├── configs/ #Hydra
│ ├── config.yaml # main Hydra config
│ ├── data/
│ │ ├── ogb_dataset.yaml # ogbl-ddi configuration
│ │ └── synthetic.yaml # synthetic dataset configuration
│ └── model/ 
│  └──  gcn.yaml #simple GCN model, new models will be added further in the process
├── src/ #Python codes
│ ├── train.py # main training loop
│ ├── train_lp.py #main training loop for synthethic dataset (link prediction)
│ ├── data/
│ │ ├── load_data.py # OGB dataset loader
│ │ ├── synthetic_dataset.py # synthetic dataset loader
│ │ └── preprocess/
│ │ └── csv_to_pyg.py # CSV → PyG transformation
│ ├── models/
│ │ ├── _init_.py # build_model() factory
│ │ └── gcn.py
│ │ └── gat.py
│ │ └── gcn_link_prediction.py
│ │ └── gin.py
│ │ └── gnn_lp.py #the most updates - GNN architecture with option to change conv_type in Hydra parameter
│ │ └── sage.py
│ └── evaluation/
│ ├── _init_.py # build_evaluator() factory
│ ├── ogb_evaluator.py # hits@20
│ └── synthetic_evaluator.py # AUPRC, ROC-AUC
├── notebooks/
│ └── train_setup_colab.ipynb # Google Colab notebook
├── data/
│ ├── raw/
│ │ └── README.md # instructions for obtaining raw data
│ └── processed/ # auto-generated (gitignored)
│ └── get_loaders.py
│ └── load_data.py
│ └── load_split_benchmark_data.py #preparation of hetero structure data
│ └── syn_transform_gnn_inputs.py #data preparation for graph classification
├── .gitignore
└── README.md
└── train_setup_colab.ipynb #training instruction in colab
```
---

## Installation

1. Copy git structure https://github.com/MoniaO/GraphNeuralNetwork_Thesis.git to your local computer
2. Use VSCode for code updates
3. Any changes push into develop branch
4. train.py - main code responsible for training, use Google colab train_setup_colab.ipynb for execute training. More details in Usage. 

## Data

### ogbl-ddi (auto-download)
Downloaded automatically via OGB on first run.

### Synthetic dataset
CSV files placed in `data/raw/` (not committed — there will be stored as WandB Artifact once we create a stable dataset).
Code for generating dataset data/raw/synthetic_dataset_generate.py

---

## Usage

Train_setup_colab.ipynb allows to execute default training via Hydra + override parameters in Hydra. 
Structure for Hydra is defined yaml files in configs/. 
Below there are included examples how training can be executed: 

```bash
# default training on ogbl-ddi
python src/train.py

# override parameters via Hydra
python src/train.py model=gcn data=ogb_dataset training.epochs=100

# synthetic dataset
python src/train.py data=synthetic model=gcn

# quick test (3 epochs)
python src/train.py training.epochs=3 training.batch_size=64
```

---

## Models

| Model | File | Description |
|---|---|---|
| GCN | `src/models/gcn.py` 

To add a new model: create a file in `src/models/`, register it in
`build_model()` in `src/models/__init__.py`, and add a config in `configs/model/`.

---

## Metrics

| Dataset | Primary metric | Additional |
|---|---|---|
| `ogbl-ddi` | Hits@20 | ROC-AUC |
| Synthetic | AUPRC | ROC-AUC, Hits@20 |

AUPRC is the primary metric for dataset with class imbalance (rare edges).
To add metrics update codes in evalution/ folder.
Metric used by specific dataset is registered in datasets yaml files in configs/data.

---

## Experiments

Results available in Weight&Bias at:
[wandb.ai/politechnika-gnn-thesis](https://wandb.ai/politechnika-gnn-thesis/politechnika-gnn-thesis)

Each run logs:
- `loss` — training loss per epoch
- `val/hits@20`, `test/hits@20`
- `val/auprc`, `val/auc` 
- `lr` — current learning rate

---

}
```
