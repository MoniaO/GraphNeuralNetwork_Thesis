"""Controlled node-feature ablations for Task A.

Profiles
--------
empirical
    Keep the original patient-derived node features.

topology_only
    Replace all node features with ones. The model can only see structure,
    node types, relation types and structural information carried by G-TRAIN.
    Input dimensionality (and therefore parameter count) is unchanged.

empirical_shuffled
    Permute complete feature rows within every node type.
    Preserves feature dimensions, marginal distributions, column means/stds
    and within-type column correlations, but destroys the assignment
    concrete node → concrete empirical feature vector.
"""

from __future__ import annotations

import hashlib
from typing import Any

import torch


SUPPORTED_FEATURE_ABLATIONS = {
    "empirical",
    "topology_only",
    "empirical_shuffled",
}


def _clone_node_features(data: Any) -> None:
    """Detach/clone feature tensors so later copies own independent storage."""
    for node_type in data.node_types:
        if not hasattr(data[node_type], "x") or data[node_type].x is None:
            raise AttributeError(
                f"Node type {node_type!r} does not contain feature tensor `x`."
            )
        data[node_type].x = data[node_type].x.detach().clone()


def apply_feature_ablation(
    data: Any,
    profile: str,
    seed: int = 20260722,
) -> Any:
    """Apply one deterministic node-feature profile in-place.

    Parameters
    ----------
    data
        PyG HeteroData with per-type `.x` tensors.
    profile
        One of: empirical | topology_only | empirical_shuffled.
    seed
        Used only by empirical_shuffled.
    """
    normalized_profile = str(profile).strip().lower()
    if normalized_profile not in SUPPORTED_FEATURE_ABLATIONS:
        raise ValueError(
            f"Unknown feature_ablation_profile={profile!r}. "
            f"Supported: {sorted(SUPPORTED_FEATURE_ABLATIONS)}"
        )

    # Important because train/valid/test are later produced as copies.
    # We want explicit ownership of feature tensors.
    _clone_node_features(data)

    if normalized_profile == "empirical":
        return data

    if normalized_profile == "topology_only":
        for node_type in sorted(data.node_types):
            original_x = data[node_type].x
            # Same shape and dtype as empirical input so architecture and
            # number of trainable parameters stay unchanged.
            data[node_type].x = torch.ones_like(original_x)
        return data

    # empirical_shuffled
    # One generator is sufficient, but permutation is per node type.
    # Drug nodes can only exchange with drug nodes, endpoints with endpoints, etc.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    for node_type in sorted(data.node_types):
        original_x = data[node_type].x
        number_of_nodes = int(original_x.size(0))
        if number_of_nodes <= 1:
            # A one-node type cannot be meaningfully permuted.
            continue

        permutation = torch.randperm(
            number_of_nodes,
            generator=generator,
            device="cpu",
        ).to(original_x.device)

        # Permute full rows, not individual cells. This preserves the
        # composition of every patient's aggregated feature vector while
        # destroying node↔feature assignment within the type.
        data[node_type].x = original_x[permutation].clone()

    return data


def feature_fingerprint(data: Any) -> str:
    """Return a stable SHA-256 fingerprint of all node-feature tensors."""
    digest = hashlib.sha256()

    for node_type in sorted(data.node_types):
        tensor = data[node_type].x.detach().cpu().contiguous()
        digest.update(node_type.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())

    return digest.hexdigest()[:16]


def feature_summary(
    data: Any,
) -> dict[str, dict[str, float | int | str]]:
    """Summarize the effective features received by the encoder."""
    summary: dict[str, dict[str, float | int | str]] = {}

    for node_type in sorted(data.node_types):
        tensor = data[node_type].x.detach().float().cpu()
        summary[node_type] = {
            "num_nodes": int(tensor.size(0)),
            "num_features": int(tensor.size(1)),
            "mean": float(tensor.mean().item()),
            "std": float(tensor.std(unbiased=False).item()),
            "minimum": float(tensor.min().item()),
            "maximum": float(tensor.max().item()),
        }

    return summary


def mean_absolute_feature_difference(
    first_data: Any,
    second_data: Any,
) -> float:
    """Mean absolute feature difference across corresponding node types."""
    first_types = sorted(first_data.node_types)
    second_types = sorted(second_data.node_types)

    if first_types != second_types:
        raise ValueError(
            "Node-type sets differ between the two HeteroData objects: "
            f"{first_types} vs {second_types}"
        )

    differences: list[torch.Tensor] = []
    for node_type in first_types:
        first_x = first_data[node_type].x.detach().float().cpu()
        second_x = second_data[node_type].x.detach().float().cpu()

        if first_x.shape != second_x.shape:
            raise ValueError(
                f"Different feature shapes for {node_type!r}: "
                f"{tuple(first_x.shape)} vs {tuple(second_x.shape)}"
            )

        differences.append(torch.abs(first_x - second_x).reshape(-1))

    if not differences:
        return 0.0

    return float(torch.cat(differences).mean().item())


def print_feature_summary(data: Any, title: str = "FEATURE SUMMARY") -> None:
    """Pretty-print per-type feature statistics."""
    print(f"\n{title}")
    for node_type, statistics in feature_summary(data).items():
        print(
            f"  {node_type}: "
            f"shape=({statistics['num_nodes']}, {statistics['num_features']}), "
            f"mean={statistics['mean']:.6f}, "
            f"std={statistics['std']:.6f}, "
            f"min={statistics['minimum']:.6f}, "
            f"max={statistics['maximum']:.6f}"
        )
