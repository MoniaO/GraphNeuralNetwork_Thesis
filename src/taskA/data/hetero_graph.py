"""HeteroData v3: typed edge stores, bez to_hetero().

Tylko zaobserwowane węzły i dodatnie krawędzie train wchodzą do message passingu.
Cechy węzłów z wierszy pacjentów train. Nie podmieniaj tu G_true.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData


# Defaults keep v2.2 filenames; pass v3 names from Hydra cfg.
NODE_FILE = "synthetic_pharmacotherapy_v2_2_nodes.csv"
EDGE_FILE = "synthetic_pharmacotherapy_v2_2_edges_audited.csv"
SAMPLE_TEMPLATE = "synthetic_pharmacotherapy_v2_2_samples_{scenario}.csv"
PATIENT_SPLIT_FILE = "patient_splits_v2_2.csv"

# Node feature ablations (Task A). Default baseline = empirical only —
# no synthetic-generator metadata (base_prevalence, observability, rarity, …).
FEATURE_PROFILES = {
    "structural": [],  # constant bias only (A0)
    "empirical": [
        "train_empirical_mean",
        "train_empirical_std",
        "train_missing_rate",
    ],
    "empirical_layer": [
        "train_empirical_mean",
        "train_empirical_std",
        "train_missing_rate",
        "layer_one_hot",
    ],
    "empirical_ontology": [
        "train_empirical_mean",
        "train_empirical_std",
        "train_missing_rate",
        "severity_weight",
        "is_endpoint",
        "layer_one_hot",
    ],
    "oracle": [
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
        "node_type_one_hot",
        "layer_one_hot",
    ],
}


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


def _empirical_stats(samples: pd.DataFrame, name: str, scenario: str) -> list[float]:
    feature_column = name
    recorded_column = f"recorded_{name}"
    if scenario == "noisy_documentation" and recorded_column in samples:
        feature_column = recorded_column
    if feature_column in samples:
        values = pd.to_numeric(samples[feature_column], errors="coerce")
        return [
            float(values.mean()),
            float(values.std(ddof=0)),
            float(values.isna().mean()),
        ]
    return [0.0, 0.0, 1.0]


def _node_features(
    nodes: pd.DataFrame,
    samples: pd.DataFrame,
    scenario: str,
    feature_profile: str = "empirical",
) -> tuple[np.ndarray, list[str]]:
    """Build per-node feature matrix according to an explicit ablation profile.

    Profiles (see FEATURE_PROFILES):
      structural         — A0: constant 1 (topology / relation types only)
      empirical          — A1: train mean/std/missing only (default, no oracle)
      empirical_layer    — A1 + DAG layer one-hot
      empirical_ontology — A1 + layer + severity + is_endpoint (no generator params)
      oracle             — old full vector incl. base_prevalence / observability / …
    """
    profile = str(feature_profile).lower().strip()
    if profile not in FEATURE_PROFILES:
        raise ValueError(
            f"Unknown node_feature_profile={feature_profile!r}. "
            f"Choose one of: {sorted(FEATURE_PROFILES)}"
        )
    selected = FEATURE_PROFILES[profile]

    node_types = sorted(nodes["node_type"].astype(str).unique())
    layers = sorted(nodes["layer"].astype(str).unique())
    rows: list[list[float]] = []
    names: list[str] = []
    standardize_flags: list[bool] = []

    # A0: constant bias so Linear/SAGE still have in_channels >= 1
    if profile == "structural":
        matrix = np.ones((len(nodes), 1), dtype=np.float32)
        return matrix, ["bias"]

    # Build name list once from the first node, then fill rows
    first = True
    for record in nodes.to_dict(orient="records"):
        parts: list[float] = []
        part_names: list[str] = []
        part_std: list[bool] = []

        empirical = _empirical_stats(samples, record["node"], scenario)
        blocks = {
            "base_prevalence": ([_number(record.get("base_prevalence", 0.0))], True),
            "severity_weight": ([_number(record.get("severity_weight", 0.0))], True),
            "observability": ([_number(record.get("observability", 0.0))], True),
            "rarity_weight": ([_number(record.get("rarity_weight", 0.0))], True),
            "node_priority_weight": ([_number(record.get("node_priority_weight", 0.0))], True),
            "is_endpoint": ([_flag(record.get("is_endpoint", False))], True),
            "is_rare_signal": ([_flag(record.get("is_rare_signal", False))], True),
            "train_empirical_mean": ([empirical[0]], True),
            "train_empirical_std": ([empirical[1]], True),
            "train_missing_rate": ([empirical[2]], True),
            "node_type_one_hot": (
                [float(record["node_type"] == value) for value in node_types],
                False,
            ),
            "layer_one_hot": (
                [float(record["layer"] == value) for value in layers],
                False,
            ),
        }
        type_names = [f"node_type={value}" for value in node_types]
        layer_names = [f"layer={value}" for value in layers]
        name_blocks = {
            "base_prevalence": ["base_prevalence"],
            "severity_weight": ["severity_weight"],
            "observability": ["observability"],
            "rarity_weight": ["rarity_weight"],
            "node_priority_weight": ["node_priority_weight"],
            "is_endpoint": ["is_endpoint"],
            "is_rare_signal": ["is_rare_signal"],
            "train_empirical_mean": ["train_empirical_mean"],
            "train_empirical_std": ["train_empirical_std"],
            "train_missing_rate": ["train_missing_rate"],
            "node_type_one_hot": type_names,
            "layer_one_hot": layer_names,
        }

        for key in selected:
            values, do_std = blocks[key]
            parts.extend(values)
            part_names.extend(name_blocks[key])
            part_std.extend([do_std] * len(values))

        if first:
            names = part_names
            standardize_flags = part_std
            first = False
        rows.append(parts)

    matrix = np.asarray(rows, dtype=np.float32)
    if matrix.size == 0:
        raise ValueError(f"Empty feature matrix for profile={profile!r}")

    continuous_idx = [i for i, flag in enumerate(standardize_flags) if flag]
    if continuous_idx:
        cols = matrix[:, continuous_idx]
        means = cols.mean(axis=0)
        stds = cols.std(axis=0)
        stds[stds < 1e-8] = 1.0
        matrix[:, continuous_idx] = (cols - means) / stds
    return matrix, names


def load_native_heterodata(
    data_dir: Path,
    scenario: str,
    train_edge_ids: set[str] | None = None,
    patient_train_fraction: float = 0.70,
    patient_split_file: Path | None = None,
    add_reverse_edges: bool = True,
    nodes_file: str = NODE_FILE,
    edges_file: str = EDGE_FILE,
    samples_file: str | None = None,
    patient_split_filename: str = PATIENT_SPLIT_FILE,
    graph_version: str = "v2.2_audited",
    feature_profile: str = "empirical",
) -> LoadedHeteroGraph:
    data_dir = Path(data_dir)
    if samples_file is None:
        samples_file = SAMPLE_TEMPLATE.format(scenario=scenario)
    nodes = pd.read_csv(data_dir / nodes_file)
    edges = pd.read_csv(data_dir / edges_file)
    nodes = nodes.loc[~_boolean(nodes["is_latent"])].reset_index(drop=True)
    observed = set(nodes["node"])
    edges = edges.loc[
        edges["source"].isin(observed) & edges["target"].isin(observed)
    ].reset_index(drop=True)
    if train_edge_ids is not None:
        edges = edges.loc[edges["edge_id"].isin(train_edge_ids)].reset_index(drop=True)

    samples = pd.read_csv(data_dir / samples_file)
    if patient_split_file is None:
        # Loader always resolves: <data_dir>/../splits/<patient_split_filename>
        # YAML should therefore use only the filename, e.g. patient_splits_v3.csv
        candidate = data_dir.parent / "splits" / patient_split_filename
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

    feature_matrix, feature_names = _node_features(
        nodes, samples, scenario, feature_profile=feature_profile
    )
    data = HeteroData()
    node_lookup: dict[str, tuple[str, int]] = {}
    for node_type, group in nodes.groupby("node_type", sort=True):
        indices = group.index.to_numpy()
        data[str(node_type)].x = torch.tensor(feature_matrix[indices], dtype=torch.float32)
        data[str(node_type)].node_name = group["node"].tolist()
        for local_index, name in enumerate(group["node"]):
            node_lookup[name] = (str(node_type), local_index)

    all_relation_keys: set[tuple[str, str, str]] = set()
    full_edges = pd.read_csv(data_dir / edges_file)
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

    data.graph_version = graph_version
    data.scenario = scenario
    data.node_feature_profile = feature_profile
    data.feature_names = feature_names
    data.patient_feature_rows = int(len(samples))
    data.add_reverse_edges = bool(add_reverse_edges)
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
