"""taskA.models.encoder — message passing na G_train.

FINAL używa wyłącznie hgt.py (HGTConv, residual, LayerNorm).
sage.py / gatv2.py / rgcn.py zostały z Stage A (wyścig backbone).

Co wolno zmieniać w NOWYM eksperymencie:
  hidden_dim, num_layers, heads, dropout — przez Hydra model.hgt.*
  wybór backbone (encoder_name) — tylko Stage A

Czego nie ruszać dla odtworzenia FINAL 14.08:
  h32, L2, heads=4, dropout=0.25, activation=leaky_relu
"""

from .base import BaseHeteroEncoder
from .factory import build_taskA_encoder
from .gatv2 import HeteroGATv2Encoder
from .hgt import HGTEncoder
from .input_projection import HeteroInputProjection
from .sage import HeteroSAGEMatchedEncoder

__all__ = [
    "BaseHeteroEncoder",
    "HeteroInputProjection",
    "HeteroSAGEMatchedEncoder",
    "HeteroGATv2Encoder",
    "HGTEncoder",
    "build_taskA_encoder",
]
