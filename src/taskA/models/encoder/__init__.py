"""taskA.models.encoder — message passing on G_train.

FINAL uses only hgt.py (HGTConv, residual, LayerNorm).
sage.py / gatv2.py / rgcn.py remain from Stage A (backbone race).

What you may change in a NEW experiment:
  hidden_dim, num_layers, heads, dropout — via Hydra model.hgt.*
  backbone choice (encoder_name) — Stage A only

What not to touch to reproduce FINAL 14.08:
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
