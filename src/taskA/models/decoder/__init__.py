"""taskA.models.decoder — scoring krawędzi A→G.

fusion88.py      gałąź grafowa 64-D + g_stat 24-D → fusion 88 → 1 logit
pair_encoder.py  MLP vs KAN na surowych 40D × 3 role → g_stat 24-D
kan_linear.py    warstwa spline (tylko twin KAN)

Co wolno zmieniać w NOWYM eksperymencie:
  stat_pair_encoder = mlp | kan_shallow
  dropout dekodera (domyślnie 0.2)
  spline_l1 dla KAN (domyślnie 1e-5)

Czego nie ruszać dla FINAL 14.08:
  GRAPH_OUT=64, STAT_DIM=24, FUSION_DIM=88
  MLP 40→16→8 i KAN 40→8 (ten sam latent, fair twin)
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
