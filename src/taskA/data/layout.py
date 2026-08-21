"""Wspólny porządek węzłów: loader indeksuje kandydatów, encoder spłaszcza embeddingi.

Nie zmieniaj tej kolejności niezależnie w loaderze i w modelu — rozjadą się indeksy par.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch


def global_layout(
    data: Any,
) -> tuple[dict[str, int], dict[str, int]]:
    """Create a stable global node-index layout.

    Node types are sorted alphabetically. Nodes inside every node type retain
    their local order from ``data[node_type].node_name``.

    Returns
    -------
    offsets
        Mapping:

            node_type -> first global index for that node type

    lookup
        Mapping:

            node_name -> global node index
    """

    offsets: dict[str, int] = {}
    lookup: dict[str, int] = {}

    offset = 0

    for node_type in sorted(data.node_types):
        node_store = data[node_type]

        if not hasattr(node_store, "node_name"):
            raise AttributeError(
                f"Node type {node_type!r} does not contain "
                "`node_name` metadata."
            )

        node_names = [
            str(name)
            for name in node_store.node_name
        ]

        number_of_nodes = int(
            node_store.num_nodes
        )

        if len(node_names) != number_of_nodes:
            raise ValueError(
                f"Node type {node_type!r} has {number_of_nodes} nodes, "
                f"but {len(node_names)} node names."
            )

        offsets[node_type] = offset

        for local_index, node_name in enumerate(node_names):
            if node_name in lookup:
                raise ValueError(
                    "Duplicated node name across node types: "
                    f"{node_name!r}."
                )

            lookup[node_name] = (
                offset
                + local_index
            )

        offset += number_of_nodes

    return offsets, lookup


def global_index_to_name(
    data: Any,
) -> list[str]:
    """Return a list mapping each global index back to its node name."""

    _, lookup = global_layout(data)

    index_to_name: list[Optional[str]] = [
        None
        for _ in range(len(lookup))
    ]

    for node_name, global_index in lookup.items():
        index_to_name[global_index] = node_name

    missing_indices = [
        index
        for index, node_name in enumerate(index_to_name)
        if node_name is None
    ]

    if missing_indices:
        raise ValueError(
            "The global node layout contains missing indices: "
            f"{missing_indices}"
        )

    return [
        str(node_name)
        for node_name in index_to_name
    ]


def flatten_embeddings(
    data: Any,
    embedding_dict: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Flatten heterogeneous embeddings using the global layout order."""

    expected_node_types = sorted(
        data.node_types
    )

    missing_node_types = [
        node_type
        for node_type in expected_node_types
        if node_type not in embedding_dict
    ]

    if missing_node_types:
        raise KeyError(
            "Embedding dictionary is missing node types: "
            f"{missing_node_types}"
        )

    embedding_dimensions = {
        int(embedding_dict[node_type].size(-1))
        for node_type in expected_node_types
    }

    if len(embedding_dimensions) != 1:
        raise ValueError(
            "All node-type embeddings must have the same dimension. "
            f"Received dimensions: {sorted(embedding_dimensions)}"
        )

    return torch.cat(
        [
            embedding_dict[node_type]
            for node_type in expected_node_types
        ],
        dim=0,
    )


def layout_signature(
    data: Any,
) -> tuple[tuple[str, int], ...]:
    """Return a stable description of node types and their sizes."""

    return tuple(
        (
            node_type,
            int(data[node_type].num_nodes),
        )
        for node_type in sorted(data.node_types)
    )


def assert_candidate_layout(
    data: Any,
    source_names: Optional[Sequence[str]] = None,
    target_names: Optional[Sequence[str]] = None,
) -> None:
    """Verify that candidate indices point to the expected node names.

    The strongest check is available when the loader stores original
    candidate names as:

        data.candidate_source_name
        data.candidate_target_name

    Alternatively, source_names and target_names can be passed directly.
    """

    if not hasattr(data, "link_source_idx"):
        raise AttributeError(
            "Data does not contain `link_source_idx`."
        )

    if not hasattr(data, "link_target_idx"):
        raise AttributeError(
            "Data does not contain `link_target_idx`."
        )

    source_indices = (
        data.link_source_idx
        .detach()
        .cpu()
        .long()
        .view(-1)
    )

    target_indices = (
        data.link_target_idx
        .detach()
        .cpu()
        .long()
        .view(-1)
    )

    if source_indices.numel() != target_indices.numel():
        raise ValueError(
            "Candidate source and target tensors have different lengths: "
            f"{source_indices.numel()} and {target_indices.numel()}."
        )

    index_to_name = global_index_to_name(
        data
    )

    number_of_nodes = len(
        index_to_name
    )

    if source_indices.numel() > 0:
        minimum_index = min(
            int(source_indices.min().item()),
            int(target_indices.min().item()),
        )

        maximum_index = max(
            int(source_indices.max().item()),
            int(target_indices.max().item()),
        )

        if minimum_index < 0:
            raise IndexError(
                f"Candidate indices contain a negative value: {minimum_index}."
            )

        if maximum_index >= number_of_nodes:
            raise IndexError(
                "Candidate index exceeds the global node layout. "
                f"Maximum candidate index: {maximum_index}; "
                f"number of nodes: {number_of_nodes}."
            )

    if source_names is None:
        source_names = getattr(
            data,
            "candidate_source_name",
            None,
        )

    if target_names is None:
        target_names = getattr(
            data,
            "candidate_target_name",
            None,
        )

    if source_names is None or target_names is None:
        # Bounds and tensor lengths were verified, but exact candidate names
        # cannot be checked without preserving the original names.
        return

    source_names = [
        str(name)
        for name in source_names
    ]

    target_names = [
        str(name)
        for name in target_names
    ]

    number_of_candidates = int(
        source_indices.numel()
    )

    if len(source_names) != number_of_candidates:
        raise ValueError(
            "Stored source names have a different length than source indices. "
            f"Names: {len(source_names)}, indices: {number_of_candidates}."
        )

    if len(target_names) != number_of_candidates:
        raise ValueError(
            "Stored target names have a different length than target indices. "
            f"Names: {len(target_names)}, indices: {number_of_candidates}."
        )

    errors: list[str] = []

    for candidate_index in range(number_of_candidates):
        source_global_index = int(
            source_indices[candidate_index].item()
        )

        target_global_index = int(
            target_indices[candidate_index].item()
        )

        decoded_source_name = index_to_name[
            source_global_index
        ]

        decoded_target_name = index_to_name[
            target_global_index
        ]

        expected_source_name = source_names[
            candidate_index
        ]

        expected_target_name = target_names[
            candidate_index
        ]

        if decoded_source_name != expected_source_name:
            errors.append(
                f"Candidate {candidate_index}: source index "
                f"{source_global_index} maps to {decoded_source_name!r}, "
                f"expected {expected_source_name!r}."
            )

        if decoded_target_name != expected_target_name:
            errors.append(
                f"Candidate {candidate_index}: target index "
                f"{target_global_index} maps to {decoded_target_name!r}, "
                f"expected {expected_target_name!r}."
            )

        if len(errors) >= 10:
            break

    if errors:
        raise AssertionError(
            "Candidate/global-layout mismatch:\n"
            + "\n".join(errors)
        )


def print_global_layout(
    data: Any,
    number_of_candidates: int = 10,
) -> None:
    """Print node-type offsets and a preview of candidate mappings."""

    offsets, _ = global_layout(
        data
    )

    index_to_name = global_index_to_name(
        data
    )

    print("\nGLOBAL NODE LAYOUT")
    print("-" * 60)

    for node_type in sorted(data.node_types):
        start = offsets[node_type]
        count = int(
            data[node_type].num_nodes
        )
        end = start + count - 1

        print(
            f"{node_type:<30} "
            f"indices {start:>4} - {end:<4} "
            f"nodes={count}"
        )

    print("-" * 60)
    print(f"Total nodes: {len(index_to_name)}")

    if not (
        hasattr(data, "link_source_idx")
        and hasattr(data, "link_target_idx")
    ):
        return

    source_indices = (
        data.link_source_idx
        .detach()
        .cpu()
        .view(-1)
    )

    target_indices = (
        data.link_target_idx
        .detach()
        .cpu()
        .view(-1)
    )

    preview_size = min(
        number_of_candidates,
        int(source_indices.numel()),
    )

    labels = getattr(
        data,
        "edge_label",
        None,
    )

    if labels is not None:
        labels = (
            labels
            .detach()
            .cpu()
            .view(-1)
        )

    print("\nCANDIDATE PREVIEW")
    print("-" * 60)

    for index in range(preview_size):
        source_index = int(
            source_indices[index].item()
        )

        target_index = int(
            target_indices[index].item()
        )

        source_name = index_to_name[
            source_index
        ]

        target_name = index_to_name[
            target_index
        ]

        if labels is None:
            label_text = "label=unknown"
        else:
            label_text = (
                f"label={int(labels[index].item())}"
            )

        print(
            f"{index:>4}: "
            f"{source_name} [{source_index}] "
            f"-> "
            f"{target_name} [{target_index}] "
            f"{label_text}"
        )
