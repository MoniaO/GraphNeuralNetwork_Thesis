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

torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])

def load_dataset(cfg: DictConfig):
    if cfg.data.name.startswith("ogb"):
        return load_ogb(cfg)
    else:
        return load_synthetic(cfg)
    
def load_ogb(cfg: DictConfig):
    dataset = PygLinkPropPredDataset(
        name=cfg.data.name,    
        root=cfg.data.root     
    )
    split_idx = dataset.get_edge_split()
    return dataset, split_idx

def load_synthetic(cfg: DictConfig):
    root = Path(cfg.data.dataset.root_dir)

    nodes = pd.read_csv(root / cfg.data.dataset.nodes_file)
    edges = pd.read_csv(root / cfg.data.dataset.edges_file)

    samples_file = cfg.data.dataset.samples[cfg.data.dataset.scenario]
    samples = pd.read_csv(root / samples_file)

    return nodes, edges, samples
    pass

