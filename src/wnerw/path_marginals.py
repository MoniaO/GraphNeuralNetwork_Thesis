"""Edge and node path-marginals from forward/backward partitions."""

from __future__ import annotations

import numpy as np

from .finite_path_ensemble import FinitePathEnsemble
from .types import NEG_INF


def edge_marginals(
    ensemble: FinitePathEnsemble,
    source: str,
    endpoint: str,
    *,
    log_forward: dict[str, float] | None = None,
    log_backward: dict[str, float] | None = None,
) -> dict[tuple[str, str], float]:
    """``P((u,v) ∈ γ | s, e)`` for every edge with positive path mass."""
    source = str(source)
    endpoint = str(endpoint)

    if log_backward is None:
        log_backward = ensemble.backward_log_partition(endpoint)
    if log_forward is None:
        log_forward = ensemble.forward_log_partition(source)

    log_z = log_backward[source]
    if log_z == NEG_INF:
        return {}

    # Consistency check: F_s(e) should equal B_e(s).
    if log_forward[endpoint] != NEG_INF:
        if not np.isclose(log_forward[endpoint], log_z, atol=1e-8, rtol=1e-8):
            raise AssertionError(
                "Forward/backward partition mismatch: "
                f"F({endpoint})={log_forward[endpoint]}, B({source})={log_z}"
            )

    marginals: dict[tuple[str, str], float] = {}
    for u, v in ensemble.graph.edges:
        fu = log_forward[u]
        bv = log_backward[v]
        if fu == NEG_INF or bv == NEG_INF:
            continue
        log_mass = fu + ensemble.edge_log_weights[(u, v)] + bv - log_z
        mass = float(np.exp(log_mass))
        if mass > 0.0:
            marginals[(u, v)] = mass

    return marginals


def node_marginals(
    ensemble: FinitePathEnsemble,
    source: str,
    endpoint: str,
    *,
    log_forward: dict[str, float] | None = None,
    log_backward: dict[str, float] | None = None,
) -> dict[str, float]:
    """``P(v ∈ γ | s, e)`` for every node on some positive-mass path."""
    source = str(source)
    endpoint = str(endpoint)

    if log_backward is None:
        log_backward = ensemble.backward_log_partition(endpoint)
    if log_forward is None:
        log_forward = ensemble.forward_log_partition(source)

    log_z = log_backward[source]
    if log_z == NEG_INF:
        return {}

    marginals: dict[str, float] = {}
    for node in ensemble.graph.nodes:
        if node == source or node == endpoint:
            # Always present on every path s→e when Z is finite.
            if log_z != NEG_INF:
                marginals[node] = 1.0
            continue

        fn = log_forward[node]
        bn = log_backward[node]
        if fn == NEG_INF or bn == NEG_INF:
            continue
        mass = float(np.exp(fn + bn - log_z))
        if mass > 0.0:
            marginals[node] = mass

    return marginals
