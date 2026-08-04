"""Wave 5 — Weighted Non-stationary Entropy Random Walk (finite-path ensemble)."""

from .calibration import PlattCalibrator, fit_platt, fit_temperature
from .evaluation import score_query_against_truth, true_paths_between
from .finite_path_ensemble import FinitePathEnsemble, logsumexp, transition_probabilities
from .gate_factors import GATES, GateSpec, gate_potential
from .graph_builder import build_wnerw_graph, layer_rank_from_nodes
from .metrics import path_entropy, path_hhi, true_path_mass
from .path_marginals import edge_marginals, node_marginals
from .potentials import build_edge_log_weights, nested_edge_log_weight, safe_log_probability
from .topk_paths import RankedPath, top_k_paths
from .types import NEG_INF

__all__ = [
    "GATES",
    "GateSpec",
    "NEG_INF",
    "FinitePathEnsemble",
    "PlattCalibrator",
    "RankedPath",
    "build_edge_log_weights",
    "build_wnerw_graph",
    "edge_marginals",
    "fit_platt",
    "fit_temperature",
    "gate_potential",
    "layer_rank_from_nodes",
    "logsumexp",
    "nested_edge_log_weight",
    "node_marginals",
    "path_entropy",
    "path_hhi",
    "safe_log_probability",
    "score_query_against_truth",
    "top_k_paths",
    "transition_probabilities",
    "true_path_mass",
    "true_paths_between",
]
