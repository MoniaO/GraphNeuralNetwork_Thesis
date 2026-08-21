"""HCR pair features used by FINAL Stage C (S10 FULL40)."""

from .binary_features import BinaryPairResult, binary_pair_features, result_to_vector
from .variable_spec import VariableSpec, VariableType
from .variable_specs_v3 import VARIABLE_SPECS

__all__ = [
    "BinaryPairResult",
    "VARIABLE_SPECS",
    "VariableSpec",
    "VariableType",
    "binary_pair_features",
    "result_to_vector",
]
