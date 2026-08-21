"""Controlled Task A interventions on HeteroData (features / relations)."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Iterable
from typing import Any

import torch


def clone_heterodata(data: Any) -> Any:
    """Deep copy to prevent one intervention from modifying another."""
    return copy.deepcopy(data)


def graph_fingerprint(data: Any) -> str:
    """Stable fingerprint of all message-passing edges."""
    digest = hashlib.sha256()

    for edge_type in sorted(data.edge_types):
        edge_index = (
            data[edge_type]
            .edge_index
            .detach()
            .cpu()
            .contiguous()
        )

        digest.update(repr(edge_type).encode("utf-8"))
        digest.update(str(tuple(edge_index.shape)).encode("utf-8"))
        digest.update(edge_index.numpy().tobytes())

    return digest.hexdigest()[:16]


def feature_fingerprint(data: Any) -> str:
    digest = hashlib.sha256()

    for node_type in sorted(data.node_types):
        x = data[node_type].x.detach().cpu().contiguous()
        digest.update(node_type.encode("utf-8"))
        digest.update(str(tuple(x.shape)).encode("utf-8"))
        digest.update(x.numpy().tobytes())

    return digest.hexdigest()[:16]


def set_all_features_to_ones(data: Any) -> Any:
    result = clone_heterodata(data)

    for node_type in result.node_types:
        result[node_type].x = torch.ones_like(result[node_type].x)

    return result


def shuffle_all_node_features(data: Any, seed: int) -> Any:
    result = clone_heterodata(data)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    for node_type in sorted(result.node_types):
        x = result[node_type].x
        if x.size(0) <= 1:
            continue

        permutation = torch.randperm(
            x.size(0),
            generator=generator,
        ).to(x.device)

        result[node_type].x = x[permutation].clone()

    return result


def shuffle_selected_node_types(
    data: Any,
    node_types: Iterable[str],
    seed: int,
) -> Any:
    result = clone_heterodata(data)
    selected = set(node_types)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    for node_type in sorted(result.node_types):
        if node_type not in selected:
            continue

        x = result[node_type].x
        if x.size(0) <= 1:
            continue

        permutation = torch.randperm(
            x.size(0),
            generator=generator,
        ).to(x.device)

        result[node_type].x = x[permutation].clone()

    return result


def remove_relation_types(
    data: Any,
    relation_names: Iterable[str],
) -> Any:
    """Remove relations from the message-passing graph.

    ``relation_names`` refers to ``edge_type[1]``, for example:
    ``risk_modifier``, ``burden_threshold``.
    Reverse edges ``rev_<name>`` are removed together with ``<name>``.
    """
    result = clone_heterodata(data)
    relation_names = set(relation_names)
    expanded = set(relation_names)
    for name in relation_names:
        expanded.add(f"rev_{name}")
        if name.startswith("rev_"):
            expanded.add(name[len("rev_") :])

    to_delete = [
        edge_type
        for edge_type in result.edge_types
        if edge_type[1] in expanded
    ]

    for edge_type in to_delete:
        del result[edge_type]

    return result


def remove_exact_edge_types(
    data: Any,
    edge_types: Iterable[tuple[str, str, str]],
) -> Any:
    result = clone_heterodata(data)

    for edge_type in edge_types:
        if edge_type in result.edge_types:
            del result[edge_type]

    return result
