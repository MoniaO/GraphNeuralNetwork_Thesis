from typing import Dict
import torch
from torch_geometric.data import HeteroData


def compute_pos_weights(train_data: HeteroData, node_to_idx: Dict[str, int]) -> Dict[int, float]:
    edge_label = train_data[("patient", "has_adr", "variable")].edge_label
    target_idx = train_data[("patient", "has_adr", "variable")].edge_label_target_idx

    weights = {}
    for t_idx in target_idx.unique().tolist():
        mask = target_idx == t_idx
        positives = edge_label[mask].sum().item()
        negatives = mask.sum().item() - positives
        weights[t_idx] = (negatives / max(positives, 1.0)) ** 0.5
    return weights