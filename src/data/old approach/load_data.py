import torch
import torch.serialization
from pathlib import Path
import pandas as pd


# monkey-patch dla PyTorch 2.6+ / OGB compatibility
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from omegaconf import DictConfig
from torch_geometric.data import DataLoader
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr, GlobalStorage
from ogb.linkproppred import PygLinkPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset
from torch.utils.data import DataLoader as TorchDataLoader
from .syn_transform_gnn_inputs import build_synthetic_graph_dataset

torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])

def load_dataset(cfg: DictConfig):
    if cfg.data.name.startswith("ogb"):
        return load_ogb(cfg)
    else:
        return build_synthetic_graph_dataset(cfg)
    
def load_ogb(cfg: DictConfig):
    dataset = PygLinkPropPredDataset(
        name=cfg.data.name,    
        root=cfg.data.root     
    )
    split_idx = dataset.get_edge_split()
    return dataset, split_idx

