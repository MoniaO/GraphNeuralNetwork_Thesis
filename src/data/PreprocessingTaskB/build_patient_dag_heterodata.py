from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
from omegaconf import DictConfig, ListConfig

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader



NODES_FILE = Path(cfg.data.dataset.root_dir) / Path(cfg.data.dataset.nodes_file) 
EDGES_FILE = Path(cfg.data.dataset.root_dir) / Path(cfg.data.dataset.edges_file)
SPLIT_FILE = Path(cfg.data.dataset.root_dir) / "splits" /  "patient_splits_v3.csv"

ENDPOINT_NODE_TYPE = ["clinical_endpoint"]

TARGET_ENDPOINTS = cfg.data.dataset.target if hasattr(cfg.data.dataset, "target") else None

ID_COLS = ["patient_id", "hospital_id"]
META_COLS = [
    "benchmark_split", "benchmark_scenario", "replicate_id",
    "generation_seed", "paired_patient_key",
]


# ---------------------------------------------------------------------------
# 1. HeteroData topology: node types, edge types, edge_index/edge_attr
# ---------------------------------------------------------------------------

def load_shared_hetero_topology(nodes_path: Path, edges_path: Path) -> dict:

    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)

    node_types = nodes["node_type"].astype(str).unique().tolist()

    # Globalny -> (node_type, lokalny_idx), potrzebny do mapowania krawedzi.
    global_to_local: Dict[str, Tuple[str, int]] = {}
    local_idx_by_type: Dict[str, Dict[str, int]] = {nt: {} for nt in node_types}
    node_names_by_type: Dict[str, List[str]] = {nt: [] for nt in node_types}

    for _, row in nodes.iterrows():
        name = str(row["node"])
        ntype = str(row["node_type"])
        local_idx = len(node_names_by_type[ntype])
        node_names_by_type[ntype].append(name)
        local_idx_by_type[ntype][name] = local_idx
        global_to_local[name] = (ntype, local_idx)

    # Statyczne cechy wezla per typ 
    static_cols = ["severity_weight", "observability", "rarity_weight", "node_priority_weight"]
    static_feats_by_type: Dict[str, torch.Tensor] = {}
    for ntype in node_types:
        sub = nodes[nodes["node_type"] == ntype].set_index("node").loc[node_names_by_type[ntype], static_cols]
        static_feats_by_type[ntype] = torch.tensor(sub.fillna(0.0).to_numpy(dtype=np.float32))

    is_endpoint_node = nodes.set_index("node")["is_endpoint"].astype(bool).to_dict()

    # Grupowanie krawedzi po relacji (src_type, edge_type, dst_type).
    missing = (set(edges["source"]) | set(edges["target"])) - set(global_to_local)
    if missing:
        raise ValueError(f"Edges reference nodes missing from nodes file: {sorted(missing)}")

    edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
    edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}

    edges = edges.copy()
    edges["src_type"] = edges["source"].map(lambda n: global_to_local[n][0])
    edges["dst_type"] = edges["target"].map(lambda n: global_to_local[n][0])
    edges["src_local"] = edges["source"].map(lambda n: global_to_local[n][1])
    edges["dst_local"] = edges["target"].map(lambda n: global_to_local[n][1])

    for (src_type, rel_type, dst_type), group in edges.groupby(["src_type", "edge_type", "dst_type"]):
        key = (src_type, rel_type, dst_type)
        src = torch.tensor(group["src_local"].to_numpy(), dtype=torch.long)
        dst = torch.tensor(group["dst_local"].to_numpy(), dtype=torch.long)
        edge_index_dict[key] = torch.stack([src, dst], dim=0)

        effect_size = group["effect_size"].fillna(0.5).to_numpy(dtype=np.float32)
        effect_sign = group["effect_sign"].fillna(1.0).to_numpy(dtype=np.float32)
        weight = (effect_size * effect_sign).reshape(-1, 1)
        edge_attr_dict[key] = torch.tensor(weight, dtype=torch.float)

    return {
        "node_types": node_types,
        "node_names_by_type": node_names_by_type,
        "local_idx_by_type": local_idx_by_type,
        "global_to_local": global_to_local,
        "static_feats_by_type": static_feats_by_type,
        "edge_index_dict": edge_index_dict,
        "edge_attr_dict": edge_attr_dict,
        "is_endpoint_node": is_endpoint_node,
    }


# ---------------------------------------------------------------------------
# 2. Budowa jednego HeteroData per pacjent
# ---------------------------------------------------------------------------

def build_patient_hetero_graphs(
    samples_df: pd.DataFrame,
    topology: dict,
    exclude_cols: List[str] = ID_COLS + META_COLS,
) -> List[HeteroData]:

    global_to_local = topology["global_to_local"]
    node_names_by_type = topology["node_names_by_type"]
    static_feats_by_type = topology["static_feats_by_type"]
    is_endpoint_node = topology["is_endpoint_node"]

    endpoint_type = ENDPOINT_NODE_TYPE
    endpoint_names = node_names_by_type.get(endpoint_type, [])
    available_endpoints = [e for e in endpoint_names if e in samples_df.columns]
    if not available_endpoints:
        raise ValueError(f"Brak kolumn typu {endpoint_type} w danych pacjentow.")

    # Kolumny-cechy = kolumny odpowiadajace nazwom wezlow (kazdego typu),
    # z wylaczeniem endpointow (idą do y, nie do x) i metadanych.
    feature_cols_by_type: Dict[str, List[str]] = {}
    for ntype, names in node_names_by_type.items():
        cols = [c for c in names if c in samples_df.columns and c not in exclude_cols]
        if ntype == endpoint_type:
            cols = []  # endpointy nigdy jako cecha wejsciowa - unikamy leakage
        feature_cols_by_type[ntype] = cols

    # Prekalkulacja wartosci numerycznych per typ.
    values_by_type: Dict[str, torch.Tensor] = {}
    col_local_idx_by_type: Dict[str, torch.Tensor] = {}
    for ntype, cols in feature_cols_by_type.items():
        if not cols:
            continue
        block = samples_df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        values_by_type[ntype] = torch.tensor(block.to_numpy(dtype=np.float32))
        col_local_idx_by_type[ntype] = torch.tensor(
            [global_to_local[c][1] for c in cols], dtype=torch.long
        )

    endpoint_local_idx = torch.tensor(
        [global_to_local[e][1] for e in available_endpoints], dtype=torch.long
    )
    y_values = torch.tensor(
        samples_df[available_endpoints].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    )

    n_patients = len(samples_df)
    patient_ids = samples_df["patient_id"].to_numpy() if "patient_id" in samples_df else np.arange(n_patients)

    edge_index_dict = topology["edge_index_dict"]
    edge_attr_dict = topology["edge_attr_dict"]

    graphs: List[HeteroData] = []
    for i in range(n_patients):
        data = HeteroData()

        for ntype, names in node_names_by_type.items():
            n_nodes = len(names)
            realized = torch.zeros(n_nodes, dtype=torch.float)
            if ntype in values_by_type:
                realized[col_local_idx_by_type[ntype]] = values_by_type[ntype][i]

            static_feats = static_feats_by_type[ntype]  # [n_nodes, static_dim]
            x = torch.cat([realized.unsqueeze(1), static_feats], dim=1)
            data[ntype].x = x
            data[ntype].num_nodes = n_nodes

        # Etykiety i maska tylko na wezlach typu clinical_endpoint.
        n_endpoint_nodes = len(node_names_by_type[endpoint_type])
        y = torch.full((n_endpoint_nodes,), float("nan"), dtype=torch.float)
        y_mask = torch.zeros(n_endpoint_nodes, dtype=torch.bool)
        y[endpoint_local_idx] = y_values[i]
        y_mask[endpoint_local_idx] = True
        data[endpoint_type].y = y
        data[endpoint_type].y_mask = y_mask

        # Wspolna topologia krawedzi + effect_size*effect_sign jako edge_attr,
        # identyczna dla kazdego pacjenta (struktura DAG jest stala).
        for key, eidx in edge_index_dict.items():
            data[key].edge_index = eidx
            data[key].edge_attr = edge_attr_dict[key]

        data.patient_id = int(patient_ids[i])
        graphs.append(data)

    return graphs


# ---------------------------------------------------------------------------
# 3. Podzial train/valid/test
# ---------------------------------------------------------------------------

def attach_splits(graphs: List[HeteroData], split_path: Path) -> Dict[str, List[HeteroData]]:
    split_df = pd.read_csv(split_path)
    split_map = dict(zip(split_df["patient_id"].astype(int), split_df["split"]))

    buckets: Dict[str, List[HeteroData]] = {"train": [], "validation": [], "test": []}
    for g in graphs:
        split = split_map.get(g.patient_id)
        if split is None:
            continue
        buckets.setdefault(split, []).append(g)
    return buckets


# ---------------------------------------------------------------------------
# 4. Pipeline glowny
# ---------------------------------------------------------------------------

def main(scenario: str = "clean", batch_size: int = 32):
    topology = load_shared_hetero_topology(NODES_FILE, EDGES_FILE)

    samples_path = Path(cfg.data.dataset.root_dir) / f"synthetic_pharmacotherapy_v3_samples_{scenario}.csv"
    samples_df = pd.read_csv(samples_path, low_memory=False)

    graphs = build_patient_hetero_graphs(samples_df, topology)
    buckets = attach_splits(graphs, SPLIT_FILE)

    loaders = {
        split: DataLoader(items, batch_size=batch_size, shuffle=(split == "train"))
        for split, items in buckets.items() if items
    }

    print(f"Scenario: {scenario}")
    print(f"Total patient hetero-graphs: {len(graphs)}")
    for split, items in buckets.items():
        print(f"  {split}: {len(items)} graphs")
    print(f"Node types: {topology['node_types']}")
    print(f"Relation types: {len(topology['edge_index_dict'])}")
    for ntype, names in topology["node_names_by_type"].items():
        print(f"  {ntype}: {len(names)} nodes")

    return loaders, topology


if __name__ == "__main__":
    main()
