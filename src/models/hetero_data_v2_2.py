#!/usr/bin/env python3
"""Native PyG HeteroData construction for the v2.2 structural benchmark.

The loader deliberately does not call ``to_hetero()``. Each audited relation is
created directly as a typed ``(source_type, relation, target_type)`` edge store.
Only observed nodes and explicitly supplied training edges enter message passing.
Patient-derived node summaries are computed from patient-training rows only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData


NODE_FILE = "synthetic_pharmacotherapy_v2_2_nodes.csv"
EDGE_FILE = "synthetic_pharmacotherapy_v2_2_edges_audited.csv"
SAMPLE_TEMPLATE = "synthetic_pharmacotherapy_v2_2_samples_{scenario}.csv"
PATIENT_SPLIT_FILE = "patient_splits_v2_2.csv"


@dataclass
class LoadedHeteroGraph:
    data: HeteroData
    nodes: pd.DataFrame
    edges: pd.DataFrame
    node_lookup: dict[str, tuple[str, int]]
    feature_names: list[str]


def _boolean(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _number(value: object) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else 0.0


def _flag(value: object) -> float:
    return float(str(value).lower() in {"true", "1", "yes"})


def _node_features(
    nodes: pd.DataFrame,
    samples: pd.DataFrame,
    scenario: str,
) -> tuple[np.ndarray, list[str]]:
    node_types = sorted(nodes["node_type"].astype(str).unique())
    layers = sorted(nodes["layer"].astype(str).unique())
    rows: list[list[float]] = []
    for record in nodes.to_dict(orient="records"):
        name = record["node"]
        feature_column = name
        recorded_column = f"recorded_{name}"
        if scenario == "noisy_documentation" and recorded_column in samples:
            feature_column = recorded_column
        if feature_column in samples:
            values = pd.to_numeric(samples[feature_column], errors="coerce")
            empirical = [
                float(values.mean()),
                float(values.std(ddof=0)),
                float(values.isna().mean()),
            ]
        else:
            empirical = [0.0, 0.0, 1.0]
        static = [
            _number(record.get("base_prevalence", 0.0)),
            _number(record.get("severity_weight", 0.0)),
            _number(record.get("observability", 0.0)),
            _number(record.get("rarity_weight", 0.0)),
            _number(record.get("node_priority_weight", 0.0)),
            _flag(record.get("is_endpoint", False)),
            _flag(record.get("is_rare_signal", False)),
        ]
        type_one_hot = [float(record["node_type"] == value) for value in node_types]
        layer_one_hot = [float(record["layer"] == value) for value in layers]
        rows.append(static + empirical + type_one_hot + layer_one_hot)

    names = [
        "base_prevalence",
        "severity_weight",
        "observability",
        "rarity_weight",
        "node_priority_weight",
        "is_endpoint",
        "is_rare_signal",
        "train_empirical_mean",
        "train_empirical_std",
        "train_missing_rate",
    ]
    names += [f"node_type={value}" for value in node_types]
    names += [f"layer={value}" for value in layers]
    matrix = np.asarray(rows, dtype=np.float32)
    continuous = np.arange(10)
    means = matrix[:, continuous].mean(axis=0)
    stds = matrix[:, continuous].std(axis=0)
    stds[stds < 1e-8] = 1.0
    matrix[:, continuous] = (matrix[:, continuous] - means) / stds
    return matrix, names


def load_native_heterodata(
    data_dir: Path,
    scenario: str,
    train_edge_ids: set[str] | None = None,
    patient_train_fraction: float = 0.70,
    patient_split_file: Path | None = None,
    add_reverse_edges: bool = True,
) -> LoadedHeteroGraph:
    data_dir = Path(data_dir)
    nodes = pd.read_csv(data_dir / NODE_FILE)
    edges = pd.read_csv(data_dir / EDGE_FILE)
    nodes = nodes.loc[~_boolean(nodes["is_latent"])].reset_index(drop=True)
    observed = set(nodes["node"])
    edges = edges.loc[
        edges["source"].isin(observed) & edges["target"].isin(observed)
    ].reset_index(drop=True)
    if train_edge_ids is not None:
        edges = edges.loc[edges["edge_id"].isin(train_edge_ids)].reset_index(drop=True)

    samples = pd.read_csv(data_dir / SAMPLE_TEMPLATE.format(scenario=scenario))
    if patient_split_file is None:
        candidate = data_dir.parent / "splits" / PATIENT_SPLIT_FILE
        patient_split_file = candidate if candidate.exists() else None
    if patient_split_file is not None:
        patient_splits = pd.read_csv(patient_split_file, dtype={"patient_id": str})
        train_patient_ids = set(
            patient_splits.loc[
                patient_splits["split"].eq("train"), "patient_id"
            ]
        )
        samples = samples.loc[
            samples["patient_id"].astype(str).isin(train_patient_ids)
        ].copy()
    else:
        patient_hash = pd.util.hash_pandas_object(
            samples["patient_id"].astype(str), index=False
        ).to_numpy(dtype=np.uint64)
        threshold = int(patient_train_fraction * 10_000)
        samples = samples.loc[(patient_hash % 10_000) < threshold].copy()

    feature_matrix, feature_names = _node_features(nodes, samples, scenario)
    data = HeteroData()
    node_lookup: dict[str, tuple[str, int]] = {}
    for node_type, group in nodes.groupby("node_type", sort=True):
        indices = group.index.to_numpy()
        data[str(node_type)].x = torch.tensor(feature_matrix[indices], dtype=torch.float32)
        data[str(node_type)].node_name = group["node"].tolist()
        for local_index, name in enumerate(group["node"]):
            node_lookup[name] = (str(node_type), local_index)

    all_relation_keys: set[tuple[str, str, str]] = set()
    full_edges = pd.read_csv(data_dir / EDGE_FILE)
    full_edges = full_edges.loc[
        full_edges["source"].isin(observed) & full_edges["target"].isin(observed)
    ]
    for row in full_edges.itertuples(index=False):
        source_type = node_lookup[row.source][0]
        target_type = node_lookup[row.target][0]
        all_relation_keys.add((source_type, str(row.edge_type), target_type))

    grouped: dict[tuple[str, str, str], list[tuple[int, int]]] = {
        key: [] for key in all_relation_keys
    }
    for row in edges.itertuples(index=False):
        source_type, source_index = node_lookup[row.source]
        target_type, target_index = node_lookup[row.target]
        key = (source_type, str(row.edge_type), target_type)
        grouped[key].append((source_index, target_index))

    for key, pairs in grouped.items():
        if pairs:
            edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        data[key].edge_index = edge_index
        if add_reverse_edges:
            reverse_key = (key[2], f"rev_{key[1]}", key[0])
            data[reverse_key].edge_index = edge_index.flip(0)

    data.graph_version = "v2.2_audited"
    data.scenario = scenario
    data.patient_feature_rows = int(len(samples))
    return LoadedHeteroGraph(
        data=data,
        nodes=nodes,
        edges=edges,
        node_lookup=node_lookup,
        feature_names=feature_names,
    )


def validate_heterodata(loaded: LoadedHeteroGraph) -> dict[str, int]:
    data = loaded.data
    expected_nodes = len(loaded.nodes)
    actual_nodes = sum(data[node_type].num_nodes for node_type in data.node_types)
    if expected_nodes != actual_nodes:
        raise AssertionError(f"Node count mismatch: {actual_nodes} != {expected_nodes}")
    expected_edges = 2 * len(loaded.edges)
    actual_edges = sum(
        data[key].edge_index.shape[1] for key in data.edge_types
    )
    if expected_edges != actual_edges:
        raise AssertionError(f"Edge count mismatch: {actual_edges} != {expected_edges}")
    for key in data.edge_types:
        edge_index = data[key].edge_index
        if edge_index.shape[0] != 2:
            raise AssertionError(f"Invalid edge_index for {key}: {edge_index.shape}")
        if edge_index.numel():
            if int(edge_index[0].max()) >= data[key[0]].num_nodes:
                raise AssertionError(f"Source index out of bounds for {key}")
            if int(edge_index[1].max()) >= data[key[2]].num_nodes:
                raise AssertionError(f"Target index out of bounds for {key}")
    return {
        "node_types": len(data.node_types),
        "edge_types_with_reverse": len(data.edge_types),
        "nodes": actual_nodes,
        "directed_message_edges": actual_edges,
        "feature_dimension": data[data.node_types[0]].x.shape[1],
        "patient_train_rows": data.patient_feature_rows,
    }
