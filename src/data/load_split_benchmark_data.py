from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import warnings
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch_geometric.data import HeteroData

DEFAULT_ENDPOINTS = [
    "AKI", "DILI", "Depression", "Falls", "Delirium", "GI_bleeding",
    "Hyponatremia", "Hyperkalemia", "QT_arrhythmia", "Hospitalization",
]

ID_COLS = ["patient_id", "hospital_id"]
META_COLS = [
    "benchmark_split", "benchmark_scenario", "replicate_id",
    "generation_seed", "paired_patient_key",
]

SPLIT_MAP = {
    "train": "train",
    "validation": "valid",
    "val": "valid",
    "valid": "valid",
    "test": "test",
    "internal_test": "test",
    "external_test": "test",
}


def _normalize_split(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip().str.lower()
    mapped = raw.map(SPLIT_MAP)
    if mapped.isna().any():
        unknown = sorted(raw[mapped.isna()].unique().tolist())
        raise ValueError(f"Unknown benchmark_split values found: {unknown}")
    return mapped


def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    first = [c for c in [
        "patient_id", "hospital_id", "benchmark_split", "benchmark_scenario",
        "replicate_id", "generation_seed", "paired_patient_key",
    ] if c in df.columns]
    rest = [c for c in df.columns if c not in first]
    return df[first + rest]


def _move_col_front(df: pd.DataFrame, col: str) -> pd.DataFrame:
    cols = [col] + [c for c in df.columns if c != col]
    return df[cols]


def _read_scenario(root: Path, scenario: str, filename: str) -> pd.DataFrame:
    path = root / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing benchmark scenario file: {path}")
    df = pd.read_csv(path, low_memory=False)
    if "benchmark_scenario" not in df.columns:
        df["benchmark_scenario"] = scenario
    return df


def _get_scenario_files(cfg: DictConfig) -> Dict[str, str]:
    dataset_cfg = cfg.data.dataset

    if "samples" in dataset_cfg and dataset_cfg.samples is not None:
        samples = {str(k): str(v) for k, v in dataset_cfg.samples.items()}
        scenario = getattr(dataset_cfg, "scenario", None)
        if scenario is not None:
            scenario = str(scenario)
            if scenario not in samples:
                raise ValueError(f"Scenario '{scenario}' not found in dataset.samples. Available: {sorted(samples.keys())}")
            return {scenario: samples[scenario]}
        return samples

    if "scenarios" in dataset_cfg and dataset_cfg.scenarios is not None:
        scenarios = {str(k): str(v) for k, v in dataset_cfg.scenarios.items()}
        scenario = getattr(dataset_cfg, "scenario", None)
        if scenario is not None:
            scenario = str(scenario)
            if scenario not in scenarios:
                raise ValueError(f"Scenario '{scenario}' not found in dataset.scenarios. Available: {sorted(scenarios.keys())}")
            return {scenario: scenarios[scenario]}
        return scenarios

    if "scenario" in dataset_cfg and dataset_cfg.scenario is not None:
        scenario_name = str(dataset_cfg.scenario)
        if "scenario_file" not in dataset_cfg:
            raise ValueError(
                "If cfg.data.dataset.scenario is used alone, cfg.data.dataset.scenario_file must also be provided."
            )
        return {scenario_name: str(dataset_cfg.scenario_file)}

    raise ValueError(
        "No scenario files configured. Use cfg.data.dataset.samples or cfg.data.dataset.scenarios."
    )


def _report_and_fill_nan_columns(feature_block: pd.DataFrame, split_name: str) -> pd.DataFrame:
    numeric_block = feature_block.apply(pd.to_numeric, errors="coerce")
    nan_counts = numeric_block.isna().sum()
    nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)

    if len(nan_counts) > 0:
        msg_lines = [
            f"NaN detected in split='{split_name}' for {len(nan_counts)} feature columns.",
            "Columns with NaN counts:",
        ]
        msg_lines.extend([f"- {col}: {int(cnt)}" for col, cnt in nan_counts.items()])
        warnings.warn("\n".join(msg_lines))

        filled_block = numeric_block.fillna(0.0)

    return numeric_block


def load_structural_graph(cfg: DictConfig) -> Dict[str, pd.DataFrame]:
    dataset_cfg = cfg.data.dataset
    root = Path(dataset_cfg.root_dir)

    nodes_file = str(getattr(dataset_cfg, "nodes_file", "synthetic_pharmacotherapy_v2_nodes.csv"))
    edges_file = getattr(dataset_cfg, "edges_file", None)

    nodes_path = root / nodes_file
    if not nodes_path.exists():
        raise FileNotFoundError(f"Missing nodes file: {nodes_path}")
    nodes = pd.read_csv(nodes_path)

    if edges_file is None:
        audited = root / "synthetic_pharmacotherapy_v2_1_edges_audited.csv"
        base = root / "synthetic_pharmacotherapy_v2_edges.csv"
        edges_path = audited if audited.exists() else base
    else:
        edges_path = root / str(edges_file)

    if not edges_path.exists():
        raise FileNotFoundError(f"Missing edges file: {edges_path}")
    edges = pd.read_csv(edges_path)

    node_names = nodes["node"].astype(str).tolist()
    node_to_idx = {node: i for i, node in enumerate(node_names)}

    node_index_df = nodes.copy()
    node_index_df["node_idx"] = node_index_df["node"].map(node_to_idx)
    node_index_df = _move_col_front(node_index_df, "node_idx")

    edge_df = edges.copy()
    edge_df = edge_df[
        edge_df["source"].isin(node_to_idx) & edge_df["target"].isin(node_to_idx)
    ].copy()
    edge_df["source_idx"] = edge_df["source"].map(node_to_idx)
    edge_df["target_idx"] = edge_df["target"].map(node_to_idx)
    edge_df = _move_col_front(edge_df, "target_idx")
    edge_df = _move_col_front(edge_df, "source_idx")

    return {
        "node_index": node_index_df.reset_index(drop=True),
        "structural_edges": edge_df.reset_index(drop=True),
        "node_to_idx": node_to_idx,
    }


def build_combined_splits(cfg: DictConfig) -> Dict[str, pd.DataFrame]:
    dataset_cfg = cfg.data.dataset
    root = Path(dataset_cfg.root_dir)
    scenario_files = _get_scenario_files(cfg)

    include_external_test = bool(getattr(dataset_cfg, "include_external_test", False))
    keep_scenario = bool(getattr(dataset_cfg, "keep_scenario", False))

    frames = []
    for scenario, filename in scenario_files.items():
        frames.append(_read_scenario(root, scenario, filename))

    combined = pd.concat(frames, ignore_index=True, sort=False)

    if "benchmark_split" not in combined.columns:
        raise ValueError(
            "Expected column 'benchmark_split' not found. "
            "Input files should come from build_v2_1_nn_benchmark.py"
        )

    raw_split = combined["benchmark_split"].astype(str).str.strip().str.lower()

    if not include_external_test:
        combined = combined.loc[raw_split != "external_test"].copy()
        raw_split = combined["benchmark_split"].astype(str).str.strip().str.lower()

    combined["benchmark_split"] = _normalize_split(raw_split)

    if not keep_scenario and "benchmark_scenario" in combined.columns:
        combined = combined.drop(columns=["benchmark_scenario"])

    combined = _reorder_columns(combined)

    return {
        "all": combined.reset_index(drop=True),
        "train": combined.loc[combined["benchmark_split"] == "train"].reset_index(drop=True),
        "valid": combined.loc[combined["benchmark_split"] == "valid"].reset_index(drop=True),
        "test": combined.loc[combined["benchmark_split"] == "test"].reset_index(drop=True),
    }


def build_patient_graph_for_split(
    df: pd.DataFrame,
    node_to_idx: Dict[str, int],
    target_endpoints: List[str],
    split_name: str,
    exclude_cols: Iterable[str] = tuple(ID_COLS) + tuple(META_COLS),
) -> Dict[str, pd.DataFrame]:
    df = df.reset_index(drop=True).copy()
    n = len(df)

    patient_index = pd.DataFrame({
        "patient_id": df["patient_id"] if "patient_id" in df.columns else np.arange(n),
        "patient_local_idx": np.arange(n),
    })

    available_targets = [e for e in target_endpoints if e in df.columns]
    if not available_targets:
        raise ValueError(
            f"None of target_endpoints={target_endpoints} are present in dataframe columns."
        )

    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols and c not in available_targets and c in node_to_idx
    ]
    if not feature_cols:
        raise ValueError("No usable feature columns found after filtering metadata and targets.")

    feature_block = _report_and_fill_nan_columns(df[feature_cols], split_name=split_name)
    feature_block = feature_block.fillna(0.0)

    node_idx_map = np.array([node_to_idx[c] for c in feature_cols])
    values = feature_block.to_numpy(dtype=float)
    patient_rep = np.repeat(np.arange(n), len(feature_cols))
    node_rep = np.tile(node_idx_map, n)
    values_flat = values.reshape(-1)

    node_features = pd.DataFrame({
        "patient_local_idx": patient_rep,
        "node_idx": node_rep,
        "value": values_flat,
    })

    binary_mask = feature_block.isin([0.0, 1.0]).to_numpy().reshape(-1)
    active_mask = (values_flat == 1.0) & binary_mask
    feature_edges = node_features.loc[active_mask].reset_index(drop=True)

    link_rows = []
    for endpoint in available_targets:
        if endpoint not in node_to_idx:
            continue
        link_rows.append(pd.DataFrame({
            "patient_local_idx": np.arange(n),
            "target_node": endpoint,
            "target_node_idx": node_to_idx[endpoint],
            "label": df[endpoint].astype(int).to_numpy(),
        }))
    link_labels = pd.concat(link_rows, ignore_index=True)

    return {
        "patient_index": patient_index,
        "node_features": node_features,
        "feature_edges": feature_edges,
        "link_labels": link_labels,
        "feature_cols": feature_cols,
    }


def build_gnn_link_prediction_inputs(cfg: DictConfig) -> Dict[str, Dict[str, pd.DataFrame]]:
    target = str(cfg.data.target)
    target_endpoints = [target]

    splits = build_combined_splits(cfg)
    structural = load_structural_graph(cfg)
    node_to_idx = structural["node_to_idx"]

    out = {
        "structural": {
            "node_index": structural["node_index"],
            "structural_edges": structural["structural_edges"],
        }
    }

    for split_name in ["train", "valid", "test"]:
        out[split_name] = build_patient_graph_for_split(
            df=splits[split_name],
            node_to_idx=node_to_idx,
            target_endpoints=target_endpoints,
            split_name=split_name,
        )
        out[split_name]["frame"] = splits[split_name]

    return out


def _build_variable_node_features(node_index_df: pd.DataFrame) -> torch.Tensor:
    node_types = node_index_df["node_type"].astype(str).fillna("unknown") if "node_type" in node_index_df.columns else pd.Series(["unknown"] * len(node_index_df))
    unique_types = sorted(node_types.unique().tolist())
    type_to_idx = {t: i for i, t in enumerate(unique_types)}

    x = torch.zeros((len(node_index_df), len(unique_types) + 1), dtype=torch.float32)
    for row_idx, node_type in enumerate(node_types):
        x[row_idx, type_to_idx[node_type]] = 1.0

    if "layer" in node_index_df.columns:
        layer_vals = pd.to_numeric(node_index_df["layer"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        x[:, -1] = torch.from_numpy(layer_vals)

    return x


def _build_patient_node_features(frame_df: pd.DataFrame, feature_cols: List[str]) -> torch.Tensor:
    if not feature_cols:
        raise ValueError("feature_cols is empty; cannot build patient node features.")
    x = frame_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    return torch.from_numpy(x)


def to_heterodata(
    split_block: Dict[str, pd.DataFrame],
    structural_block: Dict[str, pd.DataFrame],
    include_reverse_structural: bool = False,
    include_reverse_patient_has: bool = True,
) -> HeteroData:
    data = HeteroData()

    frame_df = split_block["frame"]
    patient_index_df = split_block["patient_index"]
    feature_edges_df = split_block["feature_edges"]
    link_labels_df = split_block["link_labels"]
    feature_cols = split_block["feature_cols"]

    node_index_df = structural_block["node_index"]
    structural_edges_df = structural_block["structural_edges"]

    data["patient"].num_nodes = len(patient_index_df)
    data["variable"].num_nodes = len(node_index_df)

    data["patient"].x = _build_patient_node_features(frame_df, feature_cols)
    data["variable"].x = _build_variable_node_features(node_index_df)

    structural_edge_index = torch.tensor(
        structural_edges_df[["source_idx", "target_idx"]].to_numpy().T,
        dtype=torch.long,
    )
    data[("variable", "causes", "variable")].edge_index = structural_edge_index

    if "effect_size" in structural_edges_df.columns:
        data[("variable", "causes", "variable")].edge_attr = torch.tensor(
            structural_edges_df[["effect_size"]].fillna(0.0).to_numpy(),
            dtype=torch.float32,
        )

    if include_reverse_structural:
        data[("variable", "rev_causes", "variable")].edge_index = structural_edge_index.flip(0)
        if "effect_size" in structural_edges_df.columns:
            data[("variable", "rev_causes", "variable")].edge_attr = torch.tensor(
                structural_edges_df[["effect_size"]].fillna(0.0).to_numpy(),
                dtype=torch.float32,
            )

    patient_has_edge_index = torch.tensor(
        feature_edges_df[["patient_local_idx", "node_idx"]].to_numpy().T,
        dtype=torch.long,
    ) if len(feature_edges_df) else torch.empty((2, 0), dtype=torch.long)
    data[("patient", "has", "variable")].edge_index = patient_has_edge_index

    if len(feature_edges_df):
        data[("patient", "has", "variable")].edge_attr = torch.tensor(
            feature_edges_df[["value"]].to_numpy(), dtype=torch.float32
        )

    if include_reverse_patient_has:
        data[("variable", "rev_has", "patient")].edge_index = patient_has_edge_index.flip(0)
        if len(feature_edges_df):
            data[("variable", "rev_has", "patient")].edge_attr = torch.tensor(
                feature_edges_df[["value"]].to_numpy(), dtype=torch.float32
            )

    edge_label_index = torch.tensor(
        link_labels_df[["patient_local_idx", "target_node_idx"]].to_numpy().T,
        dtype=torch.long,
    ) if len(link_labels_df) else torch.empty((2, 0), dtype=torch.long)
    edge_label = torch.tensor(
        link_labels_df["label"].to_numpy(), dtype=torch.float32
    ) if len(link_labels_df) else torch.empty((0,), dtype=torch.float32)

    data[("patient", "has_adr", "variable")].edge_label_index = edge_label_index
    data[("patient", "has_adr", "variable")].edge_label = edge_label

    if "target_node" in link_labels_df.columns and len(link_labels_df):
        data[("patient", "has_adr", "variable")].target_name = str(link_labels_df["target_node"].iloc[0])

    data["patient"].patient_id = torch.tensor(patient_index_df["patient_local_idx"].to_numpy(), dtype=torch.long)
    data["variable"].node_idx = torch.tensor(node_index_df["node_idx"].to_numpy(), dtype=torch.long)

    return data


def load_split_benchmark_data(cfg: DictConfig) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    gnn_inputs = build_gnn_link_prediction_inputs(cfg)
    return gnn_inputs["train"], gnn_inputs["valid"], gnn_inputs["test"]


def load_split_benchmark_heterodata(cfg: DictConfig) -> Tuple[HeteroData, HeteroData, HeteroData]:
    gnn_inputs = build_gnn_link_prediction_inputs(cfg)
    structural = gnn_inputs["structural"]
    train_data = to_heterodata(gnn_inputs["train"], structural)
    valid_data = to_heterodata(gnn_inputs["valid"], structural)
    test_data = to_heterodata(gnn_inputs["test"], structural)
    return train_data, valid_data, test_data


def summarize_loaded_data(cfg: DictConfig) -> pd.DataFrame:
    gnn_inputs = build_gnn_link_prediction_inputs(cfg)
    rows = []
    for split_name in ["train", "valid", "test"]:
        block = gnn_inputs[split_name]
        rows.append({
            "split": split_name,
            "patients": len(block["patient_index"]),
            "feature_edges": len(block["feature_edges"]),
            "node_feature_rows": len(block["node_features"]),
            "link_rows": len(block["link_labels"]),
            "positive_rate": round(float(block["link_labels"]["label"].mean()), 4),
        })
    return pd.DataFrame(rows)
