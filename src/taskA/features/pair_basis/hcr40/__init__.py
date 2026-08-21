"""HCR 40D pair basis used by S10 (jitter + Legendre packing)."""

from .discrete_basis import (
    audit_basis,
    fit_discrete_orthonormal_basis,
    transform_discrete_values,
)
from .encoder import Hcr40Config, Hcr40PairEncoder
from .packing import PAIR_DIM

MOTIF_DIM = 120

__all__ = [
    "PAIR_DIM",
    "MOTIF_DIM",
    "Hcr40Config",
    "Hcr40PairEncoder",
    "audit_basis",
    "fit_discrete_orthonormal_basis",
    "transform_discrete_values",
]
