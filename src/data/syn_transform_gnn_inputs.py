#!/usr/bin/env python3
"""
Prepare graph tensors for GNN modelling.

This script does not require PyTorch. It creates:
  - node_index.csv
  - edge_index.csv

You can later load edge_index.csv into PyTorch Geometric as:
  edge_index = torch.tensor(edge_index_df[["source_idx","target_idx"]].values.T, dtype=torch.long)

Patient-level GNN idea:
  - graph structure is constant for every patient,
  - every patient becomes one graph instance,
  - node feature for a given patient = value of that node in the row,
  - y = endpoint vector, e.g. AKI, DILI, Depression, Falls...
"""

"""from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=".", help="Directory with nodes/edges/sample CSV.")
    parser.add_argument("--sample", type=str, default="synthetic_pharmacotherapy_v2_samples_clean.csv")
    args = parser.parse_args()

    data_dir = Path(args.dir)
    nodes = pd.read_csv(data_dir / "synthetic_pharmacotherapy_v2_nodes.csv")
    edges = pd.read_csv(data_dir / "synthetic_pharmacotherapy_v2_edges.csv")
    sample = pd.read_csv(data_dir / args.sample, nrows=5)

    node_names = nodes["node"].tolist()
    node_index = pd.DataFrame({"node": node_names, "node_idx": range(len(node_names))})
    idx = dict(zip(node_index["node"], node_index["node_idx"]))

    edge_index = edges[["source", "target"]].copy()
    edge_index["source_idx"] = edge_index["source"].map(idx)
    edge_index["target_idx"] = edge_index["target"].map(idx)
    edge_index = edge_index.dropna(subset=["source_idx", "target_idx"])
    edge_index["source_idx"] = edge_index["source_idx"].astype(int)
    edge_index["target_idx"] = edge_index["target_idx"].astype(int)

    node_index.to_csv(data_dir / "synthetic_pharmacotherapy_v2_node_index.csv", index=False)
    edge_index.to_csv(data_dir / "synthetic_pharmacotherapy_v2_edge_index.csv", index=False)

    missing_in_sample = [n for n in node_names if n not in sample.columns]
    if missing_in_sample:
        print("Warning: some graph nodes are not present in the selected sample CSV:")
        print(missing_in_sample[:20])

    print(f"Saved node_index and edge_index to {data_dir.resolve()}")
    print("edge_index shape:", edge_index.shape)
    print(edge_index.head().to_string(index=False))


if __name__ == "__main__":
    main() """

from pathlib import Path
import pandas as pd
import torch
from torch_geometric.data import Data
from omegaconf import DictConfig
from data.load_data import load_dataset, load_synthetic

ENDPOINTS = [
    "AKI", "DILI", "Depression", "Falls", "Delirium",
    "GI_bleeding", "Hyponatremia", "Hyperkalemia",
    "QT_arrhythmia", "Hospitalization"
]

def build_synthetic_graph_dataset(cfg: DictConfig):
    nodes, edges, samples = load_synthetic(cfg)

    node_names = nodes["node"].tolist()
    node_to_idx = {node: i for i, node in enumerate(node_names)}

    edge_df = edges[["source", "target"]].copy()
    edge_df = edge_df[
        edge_df["source"].isin(node_to_idx) &
        edge_df["target"].isin(node_to_idx)
    ].copy()

    edge_df["source_idx"] = edge_df["source"].map(node_to_idx)
    edge_df["target_idx"] = edge_df["target"].map(node_to_idx)

    edge_index = torch.tensor(
        edge_df[["source_idx", "target_idx"]].values.T,
        dtype=torch.long
    )

    feature_nodes = [n for n in node_names if n in samples.columns]
    target_nodes = [t for t in ENDPOINTS if t in samples.columns]

    dataset = []

    for _, row in samples.iterrows():
        x_vals = row[feature_nodes].astype(float).values
        x = torch.tensor(x_vals, dtype=torch.float).view(-1, 1)

        y_vals = row[target_nodes].astype(float).values
        y = torch.tensor(y_vals, dtype=torch.float)

        data = Data(
            x=x,
            edge_index=edge_index,
            y=y
        )
        dataset.append(data)

    return dataset, feature_nodes, target_nodes
