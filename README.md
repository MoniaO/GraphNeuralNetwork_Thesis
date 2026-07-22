# GraphNeuralNetwork_Thesis
This thesis investigates the oversmoothing phenomenon in Graph Neural Networks 
(GNNs) applied to heterogeneous graphs with rare causal edges. We generate a 
synthetic heterogeneous graph with a controlled causal structure, enabling 
precise evaluation of model behavior under known ground truth. 
We have aim to conduct two types of experiments:
Task A: DAG graph reconstruction based on patient data (link prediction)
Task B: ADR prediction for patient based on patient data and DAG structure

## Project Overview

Experiments are conducted on two datasets:
1. **`ogbl-ddi` / `ogbl-biokg` ** — drug-drug interaction network (Open Graph Benchmark)
2. **Synthetic dataset** — medical data with rare edges and casuality

---

## Repository Structure
```
GraphNeuralNetwork_Thesis/
├── configs/ #Hydra
│ ├── config.yaml # main Hydra config
│ ├── data/
│ │ ├── dataset_proxy.yaml # ogbl-ddi configuration
│ │ └── dataset_syn.yaml # synthetic dataset configuration
│ └── model/ 
|  └── model_old_approach #models from previous run iteration
│  └── TaskA_xxx.yaml #models for task A
│  └── TaskB_xxx.yaml #modles for task B
├── data/
|  ├── raw #folder with codes to generate data
│    ├── build_v3_interaction_spec.py 
│    ├── creation_patient_splits_v3.py 
│    └── generate_synthethic_pharmacotherapy_v3_from_spec.py
├── notebooks/ #notebooks for run diagnostic & training in colab 
│ └── train_setup_colab.ipynb # Google Colab notebook
├── src/ #Python codes
│ ├── train_taskA.py # main training loop for task A
│ ├── train_taskB.py #main training loop for taskb
│ ├── data/
│   ├── PreprocessingTaskA #data prep for task A
│       ├── hetero_data_v2_2.py 
│       └── load_hetero_recon_data.py 
│   ├── PreprocessingTaskB #data prep for task B
│       └── load_split_benchmark_data.py 
│ ├── evaluation/
│ │ ├── __init__.py
│ │ ├── ogb_evaluator.py #evaluator specific for OGB dataset, not used yet
│ │ ├── oversmoothing_metrics.py #oversmoothing metrics
│ │ ├── syntetic_evalutor.py #evaluator for synthethic dataset
│ ├── models/
│   ├── __init__.py # build_model() factory
│   ├── TaskA
│   ├── TaskB
│   └── old approach.py
│ ├── training/
│   └── class_weights.py #weights for rare classes
├── .gitignore
└── README.md
```
---

## Installation

1. Copy git structure https://github.com/MoniaO/GraphNeuralNetwork_Thesis.git to your local computer
2. Use VSCode for code updates
3. Any changes push into branch task A or task b
4. train_taskX.py - main code responsible for training, use Google colab train_setup_colab.ipynb for execute training. More details in Usage. 

## Data

### ogbl-ddi/biokg 
Downloaded automatically via OGB on first run.

### Synthetic dataset
Codes for generating dataset data/raw/

---

## Usage

Train_setup_colab.ipynb allows to execute default training via Hydra + override parameters in Hydra. 
Structure for Hydra is defined yaml files in configs/. 
Below there are included examples how training can be executed: 

```bash
# default training on ogbl-ddi
python src/train.py

# override parameters via Hydra
python src/train_taskX.py model=gcn data=ogb_dataset training.epochs=100

# synthetic dataset
python src/train_taskX.py data=synthetic model=gcn

# quick test (3 epochs)
python src/train_taskX.py training.epochs=3 training.batch_size=64
```

---

## Models

To add a new model: create a file in `src/models/`, and add a config in `configs/model/`.

---

## Metrics

AUPRC is the primary metric for dataset with class imbalance (rare edges).
To add metrics update codes in evalution/ folder.
Metric used by specific dataset is registered in datasets yaml files in configs/data.

---

## Experiments

Results available in Weight&Bias at:
[wandb.ai/politechnika-gnn-thesis](https://wandb.ai/politechnika-gnn-thesis/politechnika-gnn-thesis)

---

}
```
