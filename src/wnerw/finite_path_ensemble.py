"""Exact finite-path ensemble on a DAG (backward log-partition + transitions)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import networkx as nx
import numpy as np

from .types import NEG_INF


def logsumexp(values: list[float]) -> float:
    """Numerically stable log-sum-exp over a list of log-space values."""
    if not values:
        return NEG_INF

    maximum = max(values)
    if maximum == NEG_INF:
        return NEG_INF

    return maximum + float(
        np.log(np.sum(np.exp(np.asarray(values, dtype=np.float64) - maximum)))
    )


class FinitePathEnsemble:
    """Normalized Boltzmann distribution over finite directed paths source→endpoint.

    Path weight is ``exp(S(γ) / T)`` with ``S(γ) = sum_{(u,v)∈γ} q_uv``.
    Length / motif potentials must already be folded into ``edge_log_weights``.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        edge_log_weights: Mapping[tuple[str, str], float],
        *,
        temperature: float = 1.0,
    ) -> None:
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("FinitePathEnsemble requires a DAG.")

        self.graph = graph.copy()
        self.temperature = float(temperature)

        missing = [
            (u, v)
            for u, v in self.graph.edges
            if (str(u), str(v)) not in edge_log_weights
            and (u, v) not in edge_log_weights
        ]
        if missing:
            raise KeyError(f"Missing edge log-weights for: {missing[:5]}")

        # Normalize keys to str and apply path temperature once.
        scaled: dict[tuple[str, str], float] = {}
        for u, v in self.graph.edges:
            key = (str(u), str(v))
            raw = edge_log_weights.get(key)
            if raw is None:
                raw = edge_log_weights[(u, v)]
            scaled[key] = float(raw) / self.temperature

        # Relabel graph nodes to str for a stable contract.
        mapping = {n: str(n) for n in self.graph.nodes}
        self.graph = nx.relabel_nodes(self.graph, mapping, copy=True)
        self.edge_log_weights = scaled
        self.topological_order = list(nx.topological_sort(self.graph))

    def backward_log_partition(self, endpoint: str) -> dict[str, float]:
        """``log Z_e(u)``: log total weight of finite continuations u ↝ e."""
        endpoint = str(endpoint)
        if endpoint not in self.graph:
            raise KeyError(f"Unknown endpoint: {endpoint}")

        log_partition = {node: NEG_INF for node in self.graph.nodes}
        log_partition[endpoint] = 0.0

        for source in reversed(self.topological_order):
            if source == endpoint:
                continue

            values: list[float] = []
            for target in self.graph.successors(source):
                target_partition = log_partition[target]
                if target_partition == NEG_INF:
                    continue
                values.append(
                    self.edge_log_weights[(source, target)] + target_partition
                )

            log_partition[source] = logsumexp(values)

        return log_partition

    def forward_log_partition(self, source: str) -> dict[str, float]:
        """``log F_s(v)``: log total weight of finite prefixes s ↝ v."""
        source = str(source)
        if source not in self.graph:
            raise KeyError(f"Unknown source: {source}")

        log_forward = {node: NEG_INF for node in self.graph.nodes}
        log_forward[source] = 0.0

        for node in self.topological_order:
            node_mass = log_forward[node]
            if node_mass == NEG_INF:
                continue
            for target in self.graph.successors(node):
                candidate = node_mass + self.edge_log_weights[(node, target)]
                log_forward[target] = logsumexp(
                    [log_forward[target], candidate]
                )

        return log_forward

    def log_partition_value(self, source: str, endpoint: str) -> float:
        """``log Z_{s,e}`` — total path weight from source to endpoint."""
        return self.backward_log_partition(endpoint)[str(source)]

    def path_in_graph(self, nodes: Sequence[str]) -> bool:
        nodes = [str(n) for n in nodes]
        if len(nodes) < 2:
            return False
        for u, v in zip(nodes[:-1], nodes[1:]):
            if not self.graph.has_edge(u, v):
                return False
        return True

    def path_log_weight(self, nodes: Sequence[str]) -> float:
        nodes = [str(n) for n in nodes]
        total = 0.0
        for u, v in zip(nodes[:-1], nodes[1:]):
            total += float(self.edge_log_weights[(u, v)])
        return total

    def path_probability(
        self,
        nodes: Sequence[str],
        *,
        log_partition: dict[str, float] | None = None,
    ) -> float | None:
        """Exact P(γ | s, e) under the finite ensemble, or None if γ ∉ Ω."""
        nodes = [str(n) for n in nodes]
        if not self.path_in_graph(nodes):
            return None
        source, endpoint = nodes[0], nodes[-1]
        if log_partition is None:
            log_partition = self.backward_log_partition(endpoint)
        log_z = float(log_partition[source])
        if log_z == NEG_INF:
            return None
        return float(np.exp(self.path_log_weight(nodes) - log_z))

    def count_paths(self, source: str, endpoint: str) -> int:
        """Number of directed simple paths source→endpoint on the DAG."""
        source, endpoint = str(source), str(endpoint)
        if source not in self.graph or endpoint not in self.graph:
            return 0
        counts = {n: 0 for n in self.graph.nodes}
        counts[endpoint] = 1
        for node in reversed(self.topological_order):
            if node == endpoint:
                continue
            counts[node] = sum(counts[t] for t in self.graph.successors(node))
        return int(counts[source])



def transition_probabilities(
    ensemble: FinitePathEnsemble,
    source: str,
    endpoint: str,
    log_partition: dict[str, float] | None = None,
) -> dict[str, float]:
    """Endpoint-conditioned next-step distribution ``P(v | u, e)``."""
    source = str(source)
    endpoint = str(endpoint)
    if log_partition is None:
        log_partition = ensemble.backward_log_partition(endpoint)

    source_partition = log_partition[source]
    if source_partition == NEG_INF:
        return {}

    probabilities: dict[str, float] = {}
    for target in ensemble.graph.successors(source):
        target_partition = log_partition[target]
        if target_partition == NEG_INF:
            continue
        log_probability = (
            ensemble.edge_log_weights[(source, target)]
            + target_partition
            - source_partition
        )
        probabilities[target] = float(np.exp(log_probability))

    return probabilities
