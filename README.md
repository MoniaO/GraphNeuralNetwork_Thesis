# GraphNeuralNetwork_Thesis
This thesis investigates the oversmoothing phenomenon in Graph Neural Networks 
(GNNs) applied to heterogeneous graphs with rare causal edges. We generate a 
synthetic heterogeneous graph with a controlled causal structure, enabling 
precise evaluation of model behavior under known ground truth. Using this graph, 
we benchmark selected GNN architectures on a link prediction task, with emphasis 
on three research dimensions: (1) model robustness to oversmoothing as network 
depth increases, (2) ability to detect rare causal relations under class 
imbalance, and (3) prediction stability under distribution shift caused by 
spurious correlations and varying data-generating environments.
---

## Project Overview

Experiments are conducted on two datasets:
1. **`ogbl-ddi`** — drug-drug interaction network (Open Graph Benchmark)
2. **Synthetic dataset** — medical data with rare edges

---

## Repository Structure
GraphNeuralNetwork_Thesis/
├── configs/ #Hydra
│ ├── config.yaml # main Hydra config
│ ├── data/
│ │ ├── ogb_dataset.yaml # ogbl-ddi configuration
│ │ └── synthetic.yaml # synthetic dataset configuration
│ └── model/
│ ├── gcn.yaml
│ ├── sage.yaml
│ └── gat.yaml
├── src/ #Python codes
│ ├── train.py # main training loop
│ ├── data/
│ │ ├── load_data.py # OGB dataset loader
│ │ ├── synthetic_dataset.py # synthetic dataset loader
│ │ └── preprocess/
│ │ └── csv_to_pyg.py # CSV → PyG transformation
│ ├── models/
│ │ ├── _init_.py # build_model() factory
│ │ └── gcn.py
│ └── evaluation/
│ ├── _init_.py # build_evaluator() factory
│ ├── ogb_evaluator.py # hits@20
│ └── synthetic_evaluator.py # AUPRC, ROC-AUC
├── notebooks/
│ └── train_colab.ipynb # Google Colab notebook
├── data/
│ ├── raw/
│ │ └── README.md # instructions for obtaining raw data
│ └── processed/ # auto-generated (gitignored)
├── .gitignore
└── README.md
└── train_setup_colab.ipynb #training instruction in colab


---

## Installation

1. copy git structure https://github.com/MoniaO/GraphNeuralNetwork_Thesis.git to your local computer
2. VSCode for code updates
3. Any changes push into develop branch
4. Use Google colab train_setup_colab.ipynb for training

## Data

### ogbl-ddi (auto-download)
Downloaded automatically via OGB on first run.

### Synthetic dataset
CSV files placed in `data/raw/` (not committed — there will be stored as WandB Artifact once we create a stable dataset).

To generate the PyG graph from CSV files:
```bash
python src/data/preprocess/csv_to_pyg.py
```

---

## Usage

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

**Google Colab:**
```bash
!PYTHONPATH=/content/GraphNeuralNetwork_Thesis \
  python src/train.py model=gcn data=ogb_dataset training.epochs=100
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

AUPRC is the primary metric for the synthetic dataset due to strong
class imbalance (rare edges).

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

## References

```bibtex
@inproceedings{kipf2017semi,
  title={Semi-Supervised Classification with Graph Convolutional Networks},
  author={Kipf, Thomas N. and Welling, Max},
  booktitle={ICLR},
  year={2017}
}

@inproceedings{fey2019fast,
  title={Fast Graph Representation Learning with {PyTorch Geometric}},
  author={Fey, Matthias and Lenssen, Jan E.},
  booktitle={ICLR Workshop},
  year={2019}
}

@article{hu2020open,
  title={Open Graph Benchmark: Datasets for Machine Learning on Graphs},
  author={Hu, Weihua and others},
  journal={NeurIPS},
  year={2020}
}
```
