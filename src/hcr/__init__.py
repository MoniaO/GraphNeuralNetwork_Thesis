"""Heterogeneous Clinical Relation (HCR) pair features for Task A."""

from .binary_features import BinaryPairResult, binary_pair_features, result_to_vector
from .pair_encoder import HCRPairEncoder, HCRPairEncoderConfig
from .triple_features import BinaryTripleResult, binary_triple_features, triple_result_to_vector
from .variable_spec import VariableSpec, VariableType
from .variable_specs_v3 import VARIABLE_SPECS

__all__ = [
    "BinaryPairResult",
    "BinaryTripleResult",
    "HCRPairEncoder",
    "HCRPairEncoderConfig",
    "VARIABLE_SPECS",
    "VariableSpec",
    "VariableType",
    "binary_pair_features",
    "binary_triple_features",
    "result_to_vector",
    "triple_result_to_vector",
]
