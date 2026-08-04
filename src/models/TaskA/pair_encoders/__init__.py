"""Task A pair encoders: MLP / KAN family (Wave 9 + KAN-ARCH audit)."""

from .ag_kan_residual import AGKANResidual, AGKANResidualConfig
from .factory import PairEncoderConfig, build_pair_encoder
from .grouped_kan import GroupedKANPairEncoder
from .kan import KANPairEncoder
from .kan_shallow import KANShallowPairEncoder
from .kan_to_linear import KANToLinearPairEncoder
from .linear_plus_kan import LinearPlusKANPairEncoder
from .mlp import MLPPairEncoder
from .mlp_to_kan import MLPToKANPairEncoder

__all__ = [
    "PairEncoderConfig",
    "build_pair_encoder",
    "MLPPairEncoder",
    "KANPairEncoder",
    "KANShallowPairEncoder",
    "MLPToKANPairEncoder",
    "KANToLinearPairEncoder",
    "LinearPlusKANPairEncoder",
    "GroupedKANPairEncoder",
    "AGKANResidual",
    "AGKANResidualConfig",
]
