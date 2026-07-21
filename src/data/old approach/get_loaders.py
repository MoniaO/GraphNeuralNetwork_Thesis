import torch
import torch.serialization
from pathlib import Path
import pandas as pd

from omegaconf import DictConfig
from torch_geometric.data import DataLoader
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr, GlobalStorage
from ogb.linkproppred import PygLinkPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset
from torch.utils.data import DataLoader as TorchDataLoader



def get_loaders(dataset, split_idx, cfg):
    graph = dataset[0]  

    train_edges = split_idx["train"]["edge"]   # tensor krawędzi treningowych
    val_edges   = split_idx["valid"]["edge"]

    # DataLoader iteruje po krawędziach, nie po grafach
    train_loader = TorchDataLoader(
        train_edges,
        batch_size=cfg.training.batch_size,
        shuffle=True
    )
    val_loader = TorchDataLoader(
        val_edges,
        batch_size=cfg.training.batch_size,
        shuffle=False
    )

    return graph, train_loader, val_loader