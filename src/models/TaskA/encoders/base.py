from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class BaseHeteroEncoder(torch.nn.Module, ABC):
    """Heterogeneous encoder returning per-node-type embeddings."""

    @abstractmethod
    def encode(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[Any, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward(self, data: Any) -> dict[str, torch.Tensor]:
        return self.encode(data.x_dict, data.edge_index_dict)
