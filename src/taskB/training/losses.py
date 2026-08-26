from __future__ import annotations

from typing import Callable, NamedTuple, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.loader import DataLoader


class ClassBalanceStats(NamedTuple):
    """Surowe liczniki pozytywow/wszystkich przykladow per target, policzone
    JEDNYM przejsciem po loaderze. Z tych samych liczb wyprowadza sie zarowno
    pos_weight (BCE), jak i alpha (focal loss) - bez podwajania kosztu
    przejscia po danych (ten sam blad, ktory naprawilysmy w evaluatorze przy
    select_threshold: dwa niezalezne forward-passy po tym samym loaderze)."""

    positives: torch.Tensor
    totals: torch.Tensor

    @property
    def prevalence(self) -> torch.Tensor:
        return self.positives / self.totals.clamp(min=1.0)

    def pos_weight(self, cap: float = 30.0) -> torch.Tensor:
        """Do BCEWithLogitsLoss(pos_weight=...). Nieograniczone z natury
        (neg/pos), stad cap - przy bardzo rzadkich klasach (np. DILI ~1.8%)
        bez capu waga eksplodowalaby."""
        negatives = self.totals - self.positives
        return (negatives / self.positives.clamp(min=1.0)).clamp(max=cap)

    def focal_alpha(self, min_alpha: float = 0.05, max_alpha: float = 0.95) -> torch.Tensor:
        """Do FocalLoss(alpha=...). Ograniczone do (0,1) z definicji
        (1 - prevalencja), wiec bezpieczne numerycznie nawet dla bardzo
        rzadkich klas bez potrzeby agresywnego capu jak przy pos_weight."""
        return (1.0 - self.prevalence).clamp(min=min_alpha, max=max_alpha)


def compute_class_balance_stats(
    loader: DataLoader,
    label_extractor: Callable[[object], torch.Tensor],
    n_targets: int,
    device: torch.device,
) -> ClassBalanceStats:
    """Jedno przejscie po loaderze treningowym. label_extractor(batch) musi
    zwrocic etykiety w ksztalcie doprowadzalnym do [-1, n_targets] - dzieki
    wstrzyknieciu tej funkcji modul jest task-agnostyczny (Task A i Task B
    maja rozne sposoby wyciagania etykiet z batcha, zero zaleznosci tutaj
    od konkretnego modelu/tasku)."""
    positives = torch.zeros(n_targets)
    totals = torch.zeros(n_targets)

    for batch in loader:
        batch = batch.to(device)
        labels = label_extractor(batch).view(-1, n_targets)
        positives += labels.sum(dim=0).cpu()
        totals += labels.size(0)

    return ClassBalanceStats(positives=positives, totals=totals)


class FocalLoss(nn.Module):
    """Lin i in. 2017 (RetinaNet). Drop-in zamiennik BCEWithLogitsLoss.

    FL(p_t) = -alpha_t * (1-p_t)^gamma * log(p_t)

    gamma steruje tlumieniem LATWYCH przykladow (ktore model juz dobrze
    rozpoznaje) - to jest DYNAMICZNE wazenie, zalezne od biezacej pewnosci
    modelu, w odroznieniu od alpha (stale wazenie po klasie, funkcjonalny
    odpowiednik pos_weight z BCEWithLogitsLoss).

    gamma=0 + alpha=None sprowadza sie do zwyklego BCE.
    gamma=0 + alpha z prevalencji sprowadza sie do wazonego BCE (podobne,
    ale NIE identyczne co pos_weight - inna parametryzacja tej samej idei).
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("alpha", alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = targets * p + (1 - targets) * (1 - p)
        modulating = (1 - p_t).clamp(min=1e-6).pow(self.gamma)
        loss = modulating * bce
        if self.alpha is not None:
            alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            loss = alpha_t * loss
        return loss.mean()


def build_criterion(cfg, stats: ClassBalanceStats, device: torch.device) -> nn.Module:
    """Fabryka kryterium sterowana cfg.training.loss_type ('bce' | 'focal').
    Jedno miejsce decyzji w train_taskA.py/train_taskB.py zamiast
    if/else rozsianego po petli treningowej."""
    loss_type = str(getattr(cfg.training, "loss_type", "bce")).lower()

    if loss_type == "bce":
        cap = float(getattr(cfg.training, "pos_weight_cap", 30.0))
        return nn.BCEWithLogitsLoss(pos_weight=stats.pos_weight(cap=cap).to(device))

    if loss_type == "focal":
        gamma = float(getattr(cfg.training, "focal_gamma", 2.0))
        use_alpha = bool(getattr(cfg.training, "focal_use_alpha", True))
        alpha = stats.focal_alpha().to(device) if use_alpha else None
        return FocalLoss(gamma=gamma, alpha=alpha)

    raise ValueError(f"Nieznany cfg.training.loss_type={loss_type!r}. Dostepne: 'bce', 'focal'.")
