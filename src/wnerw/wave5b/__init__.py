"""Wave 5B — Topology–Energy Decomposition and Calibration."""

from .energy_maps import EnergyConfig, build_edge_energies, uniform_edge_scores
from .exact_metrics import (
    backward_log_partition,
    exact_hub_mass,
    exact_path_entropy,
    true_path_mass,
)
from .graph_builders import (
    SelectionSpec,
    build_fixed_union_graph,
    build_model_graph,
    selected_edge_mask,
)
from .query_evaluator import evaluate_query, parse_true_path

__all__ = [
    "EnergyConfig",
    "SelectionSpec",
    "backward_log_partition",
    "build_edge_energies",
    "build_fixed_union_graph",
    "build_model_graph",
    "evaluate_query",
    "exact_hub_mass",
    "exact_path_entropy",
    "parse_true_path",
    "selected_edge_mask",
    "true_path_mass",
    "uniform_edge_scores",
]
