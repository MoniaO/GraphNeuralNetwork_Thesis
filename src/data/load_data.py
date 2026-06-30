import torch
from omegaconf import DictConfig
from torch_geometric.data import DataLoader
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr, GlobalStorage
from ogb.linkproppred import PygLinkPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset

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
    # TODO: do dodanaia pozniej
    pass


def get_loaders(dataset, split_idx, batch_size: int):
    train_loader = DataLoader(
        dataset[split_idx["train"]],
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        dataset[split_idx["valid"]],
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, val_loader