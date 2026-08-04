from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import HeteroData

# Domyslna lista 10 pierwotnych endpointow (mozna nadpisac przez cfg.data.target_endpoints).
DEFAULT_TARGET_ENDPOINTS = [
    "AKI", "DILI", "Depression", "Falls", "Delirium", "GI_bleeding",
    "Hyponatremia", "Hyperkalemia", "QT_arrhythmia", "Hospitalization",
]


class NodeClassificationHead(nn.Module):
    """Prosty MLP klasyfikujacy per-wezel. Uzywany przez KAZDA architekture
    (HeteroConv-owa i R-GCN) - dzieki temu porownanie architektur dotyczy
    wylacznie encodera, nie roznic w glowicy."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z).view(-1)


def get_targeted_labels(
    data: HeteroData,
    target_local_idx: torch.Tensor,
    target_node_type: str = "clinical_endpoint",
) -> torch.Tensor:
    """Etykiety odfiltrowane do target_endpoint_names, w tej samej
    kolejnosci co logity z forward(). Wspolne dla wszystkich architektur
    eksponujacych target_local_idx/target_node_type (Targeted, R-GCN)."""
    y = data[target_node_type].y.float()
    batch_size = int(data[target_node_type].batch.max().item()) + 1 \
        if hasattr(data[target_node_type], "batch") else 1
    n_per_graph = y.size(0) // max(batch_size, 1)
    offsets = torch.arange(batch_size, device=y.device) * n_per_graph
    idx = (offsets.unsqueeze(1) + target_local_idx.unsqueeze(0)).view(-1)
    return y[idx]


def get_targeted_wide_scores(
    data: HeteroData,
    target_local_idx: torch.Tensor,
    target_node_type: str = "clinical_endpoint",
) -> torch.Tensor:
    """"Wide" wyniki HCR (Wide&Deep) odfiltrowane do target_endpoint_names,
    w TEJ SAMEJ kolejnosci co logity z forward() i etykiety z
    get_targeted_labels() - identyczna logika indeksowania, tylko czyta
    hcr_wide zamiast y. Zawsze obecne na kazdym pacjencie (zera, jesli
    hcr_wide_df nie bylo podane przy budowie grafow - patrz
    build_patient_hetero_graphs)."""
    wide = data[target_node_type].hcr_wide.float()
    batch_size = int(data[target_node_type].batch.max().item()) + 1 \
        if hasattr(data[target_node_type], "batch") else 1
    n_per_graph = wide.size(0) // max(batch_size, 1)
    offsets = torch.arange(batch_size, device=wide.device) * n_per_graph
    idx = (offsets.unsqueeze(1) + target_local_idx.unsqueeze(0)).view(-1)
    return wide[idx]


def get_node_labels(data: HeteroData, target_node_type: str = "clinical_endpoint") -> torch.Tensor:
    """Etykiety dla WSZYSTKICH wezlow target_node_type (uzyj z
    SimplePatientDAGNodeClassifier, nie z wariantami targeted/R-GCN)."""
    return data[target_node_type].y.float()


def get_node_mask(data: HeteroData, target_node_type: str = "clinical_endpoint") -> torch.Tensor:
    """Maska wskazujaca, ktore wezly maja rzeczywista etykiete."""
    return data[target_node_type].y_mask
