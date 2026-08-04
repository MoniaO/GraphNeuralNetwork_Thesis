from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class VariableType(str, Enum):
    BINARY = "binary"
    CONTINUOUS = "continuous"
    COUNT = "count"
    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"


@dataclass(frozen=True)
class VariableSpec:
    node_id: str
    variable_type: VariableType
    column_name: str
