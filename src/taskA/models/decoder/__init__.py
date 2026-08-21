"""taskA.models.decoder — A→G edge scoring.

fusion88.py      graph branch 64-D + g_stat 24-D → fusion 88 → 1 logit
pair_encoder.py  MLP vs KAN on raw 40D × 3 roles → g_stat 24-D
kan_linear.py    spline layer (KAN twin only)

What you may change in a NEW experiment:
  stat_pair_encoder = mlp | kan_shallow
  decoder dropout (default 0.2)
  spline_l1 for KAN (default 1e-5)

What not to touch for FINAL 14.08:
  GRAPH_OUT=64, STAT_DIM=24, FUSION_DIM=88
  MLP 40→16→8 and KAN 40→8 (same latent, fair twin)
"""

from .fusion88 import FUSION_DIM, Fusion88Decoder, build_fusion88_decoder
from .kan_linear import KANLinear
from .pair_encoder import Fusion88StatDecoder, build_stage_c_decoder

__all__ = [
    "FUSION_DIM",
    "Fusion88Decoder",
    "Fusion88StatDecoder",
    "KANLinear",
    "build_fusion88_decoder",
    "build_stage_c_decoder",
]
