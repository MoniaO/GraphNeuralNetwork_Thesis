"""Wave 7 Panel B — jitter + Legendre GHCR (40-d pairs → 120-d motif)."""

from .encoder import PanelBConfig, PanelBPairEncoder
from .packing import PAIR_DIM

MOTIF_DIM = 120

__all__ = [
    "PAIR_DIM",
    "MOTIF_DIM",
    "PanelBConfig",
    "PanelBPairEncoder",
]
