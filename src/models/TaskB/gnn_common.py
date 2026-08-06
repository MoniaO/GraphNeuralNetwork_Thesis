# from __future__ import annotations

# import torch
# from torch import nn
# from torch_geometric.data import HeteroData

# # Domyslna lista 10 pierwotnych endpointow (mozna nadpisac przez cfg.data.target_endpoints).
# DEFAULT_TARGET_ENDPOINTS = [
#     "AKI", "DILI", "Depression", "Falls", "Delirium", "GI_bleeding",
#     "Hyponatremia", "Hyperkalemia", "QT_arrhythmia", "Hospitalization",
# ]


# class NodeClassificationHead(nn.Module):
#     #MLP klasyfikujacy per-wezel

#     def __init__(self, hidden_dim: int, dropout: float = 0.0):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, 1),
#         )

#     def forward(self, z: torch.Tensor) -> torch.Tensor:
#         return self.mlp(z).view(-1)


# def get_targeted_labels(
#     data: HeteroData,
#     target_local_idx: torch.Tensor,
#     target_node_type: str = "clinical_endpoint",
# ) -> torch.Tensor:
#     #Etykiety odfiltrowane do target_endpoint_names
#     y = data[target_node_type].y.float()
#     batch_size = int(data[target_node_type].batch.max().item()) + 1 \
#         if hasattr(data[target_node_type], "batch") else 1
#     n_per_graph = y.size(0) // max(batch_size, 1)
#     offsets = torch.arange(batch_size, device=y.device) * n_per_graph
#     idx = (offsets.unsqueeze(1) + target_local_idx.unsqueeze(0)).view(-1)
#     return y[idx]


# def get_targeted_wide_scores(
#     data: HeteroData,
#     target_local_idx: torch.Tensor,
#     target_node_type: str = "clinical_endpoint",
# ) -> torch.Tensor:
#     """"Wide" wyniki HCR (Wide&Deep) odfiltrowane do target_endpoint_names"""
#     wide = data[target_node_type].hcr_wide.float()
#     batch_size = int(data[target_node_type].batch.max().item()) + 1 \
#         if hasattr(data[target_node_type], "batch") else 1
#     n_per_graph = wide.size(0) // max(batch_size, 1)
#     offsets = torch.arange(batch_size, device=wide.device) * n_per_graph
#     idx = (offsets.unsqueeze(1) + target_local_idx.unsqueeze(0)).view(-1)
#     return wide[idx]


# def get_node_labels(data: HeteroData, target_node_type: str = "clinical_endpoint") -> torch.Tensor:
#     #Etykiety dla WSZYSTKICH wezlow target_node_type 
#     return data[target_node_type].y.float()


# def get_node_mask(data: HeteroData, target_node_type: str = "clinical_endpoint") -> torch.Tensor:
#     #Maska wskazujaca
#     return data[target_node_type].y_mask


from __future__ import annotations

from typing import Tuple

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


def get_targeted_pair_evidence(
    data: HeteroData,
    target_local_idx: torch.Tensor,
    target_node_type: str = "clinical_endpoint",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Wektory dowodowe HCR (9 kanalow x max_parents, patrz
    hcr_wide_features.compute_hcr_pair_evidence / PAIR_EVIDENCE_CHANNELS)
    odfiltrowane do target_endpoint_names - TA SAMA logika indeksowania co
    get_targeted_labels/get_targeted_wide_scores, tylko czyta hcr_pair_features
    (i towarzyszaca maske hcr_pair_mask). PyG batchuje te atrybuty tak samo
    jak "x"/"y" - wzdluz wymiaru wezlow, reszta ksztaltu (max_parents, 9)
    przechodzi bez zmian - stad ten sam wzor na offsety/idx dziala bez modyfikacji.

    Zwraca:
        features: [batch_size * n_targets, max_parents, 9]
        mask:     [batch_size * n_targets, max_parents]
    """
    features = data[target_node_type].hcr_pair_features.float()
    mask = data[target_node_type].hcr_pair_mask.float()
    batch_size = int(data[target_node_type].batch.max().item()) + 1 \
        if hasattr(data[target_node_type], "batch") else 1
    n_per_graph = features.size(0) // max(batch_size, 1)
    offsets = torch.arange(batch_size, device=features.device) * n_per_graph
    idx = (offsets.unsqueeze(1) + target_local_idx.unsqueeze(0)).view(-1)
    return features[idx], mask[idx]


class HCRPairEncoder(nn.Module):
    """Maly, WSPOLDZIELONY nieliniowy koder pary (rodzic, endpoint) - wersja
    "E3+E4" HCR, w miejsce pojedynczej, liniowo wazonej liczby s_p,e (ta
    pozostaje dostepna jako prostszy wariant - patrz get_targeted_wide_scores).

    Uzasadnienie: glowny wniosek architektoniczny Task A (dokument HCR,
    sekcja 7.1) - "encode each statistical pair nonlinearly before mixing
    it with other pair roles". Plaska konkatenacja/suma PRZED nieliniowoscia
    (ich wariant B2) dala wynik GORSZY niz brak HCR w ogole (0.791 vs
    baseline 0.804); nieliniowy koder per para (R1->A0) byl najwiekszym
    pojedynczym skokiem w calej ich ablacji (+0.042).

    Wejscie ma DOWOLNA liczbe wymiarow wiodacych (typowo
    [batch*n_targets, max_parents, N_PAIR_EVIDENCE_CHANNELS]) - kazdy "slot"
    rodzica kodowany TA SAMA siecia (wagi dzielone miedzy rodzicami i miedzy
    endpointami, jak w Task A pair encoder), agregowany MASKOWANA SREDNIA po
    rodzicach (padding wykluczony - inaczej endpointy z mniejsza liczba
    rodzicow dostalyby sztucznie zanizony sygnal), na koniec maly liniowy
    head -> skalar dodawany do logitu (Wide&Deep, jak w linear wariancie)."""

    def __init__(self, in_channels: int, hidden_dim: int = 8, embed_dim: int = 4, dropout: float = 0.0):
        super().__init__()
        self.pair_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.out_head = nn.Linear(embed_dim, 1)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        features: [..., max_parents, in_channels]
        mask:     [..., max_parents] (1.0 = prawdziwy rodzic, 0.0 = padding)
        zwraca:   [...] (skalar per pozycja wiodaca, np. per pacjent-endpoint)
        """
        encoded = self.pair_mlp(features)             # [..., max_parents, embed_dim]
        mask_exp = mask.unsqueeze(-1)                  # [..., max_parents, 1]
        summed = (encoded * mask_exp).sum(dim=-2)      # [..., embed_dim]
        # clamp(min=1.0): endpoint bez ZADNEGO rodzica (mask same zera) nie
        # dzieli przez 0 - dostaje wynik 0.0 (neutralny wklad do logitu),
        # zamiast NaN/Inf.
        count = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)  # [..., 1]
        pooled = summed / count                        # masked mean [..., embed_dim]
        return self.out_head(pooled).squeeze(-1)        # [...]


def get_node_labels(data: HeteroData, target_node_type: str = "clinical_endpoint") -> torch.Tensor:
    """Etykiety dla WSZYSTKICH wezlow target_node_type (uzyj z
    SimplePatientDAGNodeClassifier, nie z wariantami targeted/R-GCN)."""
    return data[target_node_type].y.float()


def get_node_mask(data: HeteroData, target_node_type: str = "clinical_endpoint") -> torch.Tensor:
    """Maska wskazujaca, ktore wezly maja rzeczywista etykiete."""
    return data[target_node_type].y_mask
