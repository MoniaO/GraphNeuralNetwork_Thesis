"""Gate-centric patient queries for Wave 5C (hidden A→G edges)."""

from __future__ import annotations

import json
from typing import Any

import networkx as nx
import pandas as pd

from hcr.motif_registry_v3 import DUAL_GATES, GATE_OUTCOMES, TRIPLE_GATES
from hcr.motifs import GATE_DEFINITIONS, OLD_GATES


def _all_gates() -> dict[str, list[str]]:
    merged = {**GATE_DEFINITIONS, **OLD_GATES, **DUAL_GATES, **TRIPLE_GATES}
    return {str(g): [str(p) for p in parents] for g, parents in merged.items()}


def build_gate_queries(
    graph: nx.DiGraph,
    context_map: dict[tuple[str, str], tuple[str, ...]],
    *,
    query_split_seed: int = 20260722,
) -> pd.DataFrame:
    """One query per (hidden parent → gate) with outcome Y when path exists in G*."""
    rows = []
    qid = 0
    gates = _all_gates()
    for gate, parents in sorted(gates.items()):
        y = GATE_OUTCOMES.get(gate)
        if y is None:
            continue
        if gate not in graph or y not in graph:
            continue
        if not graph.has_edge(gate, y) and not nx.has_path(graph, gate, y):
            continue
        for parent in parents:
            if parent not in graph:
                continue
            if not graph.has_edge(parent, gate):
                continue
            # Prefer short true path parent → gate → … → Y
            try:
                paths = list(nx.all_simple_paths(graph, parent, y, cutoff=8))
            except Exception:
                paths = []
            if not paths:
                continue
            # Prefer paths that contain the gate; require competing routes (n>1)
            # so HEM can move under gate activation.
            paths_via_gate = [p for p in paths if gate in p]
            if not paths_via_gate:
                continue
            if len(paths) < 2:
                continue
            path = sorted(paths_via_gate, key=lambda p: (len(p), p))[0]
            ctx = context_map.get(
                (parent, gate), tuple(p for p in parents if p != parent)
            )
            if not ctx:
                continue
            if (parent, gate) not in context_map:
                # Ensure energy can apply the AND bonus on this hidden edge.
                continue
            rows.append(
                {
                    "query_id": f"g{qid:04d}",
                    "source": parent,
                    "endpoint": y,
                    "gate": gate,
                    "hidden_edge_source": parent,
                    "hidden_edge_target": gate,
                    "true_path": " > ".join(path),
                    "context_nodes": json.dumps(list(ctx)),
                    "n_context": len(ctx),
                    "n_paths_in_gstar": len(paths),
                }
            )
            qid += 1

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    # Deterministic valid/test split
    rng = __import__("numpy").random.default_rng(query_split_seed)
    order = rng.permutation(len(df))
    split = ["valid"] * len(df)
    for i in order[: len(df) // 2]:
        split[int(i)] = "test"
    df["query_split"] = split
    return df


def parse_context_nodes(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(x) for x in value)
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("["):
            import json

            return tuple(str(x) for x in json.loads(s))
        if not s:
            return ()
        return tuple(p.strip() for p in s.split(",") if p.strip())
    return ()
