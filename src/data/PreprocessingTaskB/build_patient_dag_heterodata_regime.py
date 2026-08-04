from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader


ENDPOINT_NODE_TYPE = "clinical_endpoint"

ID_COLS = ["patient_id", "hospital_id"]
META_COLS = [
    "benchmark_split", "benchmark_scenario", "replicate_id",
    "generation_seed", "paired_patient_key",
]

# Kolumny brzegowych atrybutow krawedzi z pliku audit (poza effect_size/effect_sign,
# ktore zawsze wchodza jako pierwsza kolumna edge_attr).
EDGE_ATTR_NUMERIC_COLS = ["activation_frequency", "mechanistic_confidence", "evidence_weight"]

# Kolumny statycznych cech wezla. "layer" jest tekstowe w CSV (np. "3_mechanisms"),
# wiec jest konwertowane na numer warstwy topologicznej osobno (patrz _layer_to_numeric).
STATIC_FEATURE_COLS = ["severity_weight", "observability", "rarity_weight", "node_priority_weight"]

# Reżimy obserwowalności: ktore typy wezlow sa widoczne jako cecha wejsciowa (x[:,0]),
# poza typami zawsze wykluczonymi (clinical_endpoint - to etykiety, nie cechy).
# "full"              - wszystko widoczne (obecny, domyslny baseline).
# "mechanisms_latent"  - mechanizmy posrednie (typu 'mechanism') sa ukryte; model
#                        musi je zrekonstruowac z lekow/kontekstu pacjenta przez 2-3 hopy.
# "bedside"            - widoczny tylko kontekst pacjenta i ekspozycje lekowe; to,
#                        co lekarz ma przy przyjeciu, zanim przyjda wyniki badan.
REGIME_VISIBLE_TYPES: Dict[str, set] = {
    "full": {
        "patient_context", "drug_exposure", "mechanism",
        "adr_or_intermediate_state", "observation_or_selection",
    },
    "mechanisms_latent": {
        "patient_context", "drug_exposure",
        "adr_or_intermediate_state", "observation_or_selection",
    },
    "bedside": {"patient_context", "drug_exposure"},
}


def _layer_to_numeric(layer_value: object) -> float:
    """'3_mechanisms' -> 3.0. Brak/zla wartosc -> NaN (dofillowane pozniej medianą)."""
    try:
        return float(str(layer_value).split("_", 1)[0])
    except (ValueError, IndexError):
        return float("nan")


LAYER_ENCODINGS = {"numeric", "onehot"}


def _build_layer_columns(nodes: pd.DataFrame, layer_encoding: str) -> Tuple[pd.DataFrame, List[str]]:
    #Dokleja do `nodes` kolumny reprezentujace warstwe topologiczna DAG-a,
    if layer_encoding not in LAYER_ENCODINGS:
        raise ValueError(f"layer_encoding musi byc w {LAYER_ENCODINGS}, otrzymano {layer_encoding!r}.")

    nodes = nodes.copy()
    if layer_encoding == "numeric":
        nodes["layer_numeric"] = nodes["layer"].map(_layer_to_numeric)
        return nodes, ["layer_numeric"]

    # onehot
    layer_categories = sorted(nodes["layer"].astype(str).unique().tolist())
    onehot = pd.get_dummies(nodes["layer"].astype(str)).reindex(columns=layer_categories, fill_value=0)
    onehot = onehot.astype(np.float32)
    onehot.index = nodes.index
    nodes = pd.concat([nodes, onehot], axis=1)
    return nodes, layer_categories


# ---------------------------------------------------------------------------
# 1. HeteroData topology: node types, edge types, edge_index/edge_attr
# ---------------------------------------------------------------------------

def load_shared_hetero_topology(cfg) -> dict:
    dataset_cfg = cfg.data.dataset
    root = Path(dataset_cfg.root_dir)

    nodes_file = root / str(getattr(dataset_cfg, "nodes_file", "synthetic_pharmacotherapy_v3_nodes.csv"))
    edges_file = root / str(
        getattr(dataset_cfg, "edges_file", "synthetic_pharmacotherapy_v3_edges_audited.csv")
    )
    splits_file = root / f"splits/{getattr(dataset_cfg, 'splits_file', 'patient_splits_v3.csv')}"

    nodes = pd.read_csv(nodes_file)
    edges = pd.read_csv(edges_file)

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

    # --- (D) warstwa topologiczna, dolaczana do cech statycznych cfg.data.dataset.layer_encoding 
    layer_encoding = str(getattr(dataset_cfg, "layer_encoding", "numeric"))
    nodes, layer_cols = _build_layer_columns(nodes, layer_encoding)
    static_cols = STATIC_FEATURE_COLS + layer_cols

    # --- (E) lokalne indeksy wezlow, do embeddingu tozsamosci w modelu ---
    node_local_idx_by_type: Dict[str, torch.Tensor] = {
        ntype: torch.arange(len(names), dtype=torch.long)
        for ntype, names in node_names_by_type.items()
    }

    static_feats_by_type: Dict[str, torch.Tensor] = {}
    for ntype in node_types:
        sub = nodes[nodes["node_type"] == ntype].set_index("node").loc[node_names_by_type[ntype], static_cols]
        sub = sub.fillna(sub.median(numeric_only=True))
        sub = sub.fillna(0.0)  # gdyby cala kolumna byla NaN dla danego typu
        static_feats_by_type[ntype] = torch.tensor(sub.to_numpy(dtype=np.float32))

    is_endpoint_node = nodes.set_index("node")["is_endpoint"].astype(bool).to_dict()

    # --- (C) wezly wylaczone z obserwowanego modelu: ukryte konfoundery / is_latent ---
    excluded_latent = set(
        nodes.loc[
            (nodes["recommended_use"] == "exclude_from_observed_model")
            | nodes["is_latent"].astype(bool),
            "node",
        ]
    )

    # ---  wezly bedace potomkami endpointow (leakage strukturalny) - wykluczenie ---
    endpoint_names_raw = set(nodes.loc[nodes["is_endpoint"].astype(bool), "node"])
    dag_for_descendants = nx.from_pandas_edgelist(
        edges[["source", "target"]], "source", "target", create_using=nx.DiGraph
    )
    dag_for_descendants.add_nodes_from(nodes["node"])
    endpoint_descendants: set = set()
    for ep in endpoint_names_raw:
        endpoint_descendants |= nx.descendants(dag_for_descendants, ep)
    endpoint_descendants -= endpoint_names_raw  # endpointy same siebie nie wykluczaja

    excluded_nodes = excluded_latent | endpoint_descendants

    _leak_only = endpoint_descendants - excluded_latent
    if _leak_only:
        print(
            f"[build_patient_dag_heterodata] Wykluczono {len(_leak_only)} wezlow "
            f"jako potomkow endpointow (leakage): {sorted(_leak_only)}"
        )

    # (src_type, dst_type) zamiast (src_type, edge_type, dst_type) 
    missing = (set(edges["source"]) | set(edges["target"])) - set(global_to_local)
    if missing:
        raise ValueError(f"Edges reference nodes missing from nodes file: {sorted(missing)}")

    edges = edges.copy()
    edges["src_type"] = edges["source"].map(lambda n: global_to_local[n][0])
    edges["dst_type"] = edges["target"].map(lambda n: global_to_local[n][0])
    edges["src_local"] = edges["source"].map(lambda n: global_to_local[n][1])
    edges["dst_local"] = edges["target"].map(lambda n: global_to_local[n][1])

    edge_type_categories = sorted(edges["edge_type"].astype(str).unique().tolist())
    edge_type_to_idx = {name: i for i, name in enumerate(edge_type_categories)}
    edge_attr_dim = 1 + len(EDGE_ATTR_NUMERIC_COLS) + len(edge_type_categories)

    edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
    edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}

    for (src_type, dst_type), group in edges.groupby(["src_type", "dst_type"]):
        key = (src_type, "to", dst_type)
        src = torch.tensor(group["src_local"].to_numpy(), dtype=torch.long)
        dst = torch.tensor(group["dst_local"].to_numpy(), dtype=torch.long)
        edge_index_dict[key] = torch.stack([src, dst], dim=0)

        effect_size = group["effect_size"].fillna(0.5).to_numpy(dtype=np.float32)
        effect_sign = group["effect_sign"].fillna(1.0).to_numpy(dtype=np.float32)
        weight = (effect_size * effect_sign).reshape(-1, 1)

        numeric_extra = [
            group[col].fillna(group[col].median()).fillna(0.5).to_numpy(dtype=np.float32).reshape(-1, 1)
            for col in EDGE_ATTR_NUMERIC_COLS
        ]

        onehot = np.zeros((len(group), len(edge_type_categories)), dtype=np.float32)
        type_idx = group["edge_type"].astype(str).map(edge_type_to_idx).to_numpy()
        onehot[np.arange(len(group)), type_idx] = 1.0

        edge_attr = np.concatenate([weight] + numeric_extra + [onehot], axis=1)
        edge_attr_dict[key] = torch.tensor(edge_attr, dtype=torch.float)

    return {
        "node_types": node_types,
        "node_names_by_type": node_names_by_type,
        "local_idx_by_type": local_idx_by_type,
        "node_local_idx_by_type": node_local_idx_by_type,
        "global_to_local": global_to_local,
        "static_feats_by_type": static_feats_by_type,
        "edge_index_dict": edge_index_dict,
        "edge_attr_dict": edge_attr_dict,
        "edge_type_categories": edge_type_categories,
        "edge_attr_dim": edge_attr_dim,
        "is_endpoint_node": is_endpoint_node,
        "excluded_nodes": excluded_nodes,
        "splits_file": splits_file,
        "layer_encoding": layer_encoding,
        "layer_cols": layer_cols,
    }


# ---------------------------------------------------------------------------
# 1b. Statystyki normalizacyjne (liczone WYLACZNIE na splicie train)
# ---------------------------------------------------------------------------

def compute_norm_stats(
    samples_df: pd.DataFrame,
    node_names_by_type: Dict[str, List[str]],
    train_patient_ids: set,
    exclude_cols: List[str],
) -> Dict[str, Tuple[float, float]]:
    """Srednia/std per kolumna-wezel, liczone tylko na pacjentach treningowych.

    Zwraca surowy slownik dla WSZYSTKICH kolumn odpowiadajacych nazwom wezlow
    (niezaleznie od reżimu obserwowalności) - build_patient_hetero_graphs uzyje
    tylko tych wpisow, ktore faktycznie sa widoczne w danym reżimie.
    """
    if "patient_id" not in samples_df.columns:
        raise ValueError("samples_df musi zawierac kolumne 'patient_id' do wyznaczenia splitu train.")

    train_mask = samples_df["patient_id"].isin(train_patient_ids)
    train_df = samples_df.loc[train_mask]
    if train_df.empty:
        raise ValueError("Zaden wiersz samples_df nie nalezy do splitu train - sprawdz split_path/scenario.")

    stats: Dict[str, Tuple[float, float]] = {}
    all_node_names = [name for names in node_names_by_type.values() for name in names]
    for name in all_node_names:
        if name not in train_df.columns or name in exclude_cols:
            continue
        values = pd.to_numeric(train_df[name], errors="coerce")
        mean = float(values.mean())
        std = float(values.std())
        if not np.isfinite(mean):
            mean = 0.0
        if not np.isfinite(std) or std < 1e-8:
            std = 1.0
        stats[name] = (mean, std)
    return stats


# ---------------------------------------------------------------------------
# 2. Budowa jednego HeteroData per pacjent
# ---------------------------------------------------------------------------

def build_patient_hetero_graphs(
    samples_df: pd.DataFrame,
    topology: dict,
    observability_regime: str = "full",
    norm_stats: Optional[Dict[str, Tuple[float, float]]] = None,
    exclude_cols: Optional[List[str]] = None,
    hcr_wide_df: Optional[pd.DataFrame] = None,
) -> List[HeteroData]:
    """
    hcr_wide_df: opcjonalny DataFrame z compute_hcr_wide_scores (patient_id +
        kolumny endpointow -> wartosc s_p,e, "wide" sciezka Wide&Deep HCR).
        Gdy None (domyslnie) - kazdy pacjent dostaje wektor zer o dlugosci
        n_endpoint_nodes; model moze wtedy miec use_hcr_wide=True bez zadnego
        efektu (bezpieczny no-op), albo use_hcr_wide=False (ignoruje w ogole).
        Atrybut hcr_wide jest DOLACZANY ZAWSZE (nawet gdy hcr_wide_df=None),
        zeby PyG batching mial spojna strukture atrybutow miedzy grafami w
        batchu (niespojna obecnosc atrybutu miedzy pacjentami w tym samym
        batchu jest cichym zrodlem bledow przy kolacji).
    """
    if observability_regime not in REGIME_VISIBLE_TYPES:
        raise ValueError(
            f"Nieznany observability_regime={observability_regime!r}. "
            f"Dostepne: {sorted(REGIME_VISIBLE_TYPES)}"
        )
    exclude_cols = list(exclude_cols) if exclude_cols is not None else list(ID_COLS + META_COLS)

    global_to_local = topology["global_to_local"]
    node_names_by_type = topology["node_names_by_type"]
    static_feats_by_type = topology["static_feats_by_type"]
    node_local_idx_by_type = topology["node_local_idx_by_type"]
    excluded_nodes = topology["excluded_nodes"]

    endpoint_type = ENDPOINT_NODE_TYPE
    endpoint_names = node_names_by_type.get(endpoint_type, [])
    available_endpoints = [e for e in endpoint_names if e in samples_df.columns]
    if not available_endpoints:
        raise ValueError(f"Brak kolumn typu {endpoint_type} w danych pacjentow.")

    visible_types = REGIME_VISIBLE_TYPES[observability_regime]

    # --- (C) + (H) filtr: wykluczone wezly (ukryte konfoundery) + reżim obserwowalności ---
    feature_cols_by_type: Dict[str, List[str]] = {}
    for ntype, names in node_names_by_type.items():
        cols = [
            c for c in names
            if c in samples_df.columns and c not in exclude_cols and c not in excluded_nodes
        ]
        if ntype == endpoint_type or ntype not in visible_types:
            cols = []  # endpointy nigdy jako cecha; typy spoza reżimu - ukryte
        feature_cols_by_type[ntype] = cols

    # Prekalkulacja wartosci numerycznych per typ, ze standaryzacja (F) liczona na train.
    values_by_type: Dict[str, torch.Tensor] = {}
    col_local_idx_by_type: Dict[str, torch.Tensor] = {}
    for ntype, cols in feature_cols_by_type.items():
        if not cols:
            continue
        block = samples_df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        if norm_stats is not None:
            for c in cols:
                mean, std = norm_stats.get(c, (0.0, 1.0))
                block[c] = (block[c] - mean) / std
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

    # --- HCR "wide" (Wide&Deep)
    n_endpoint_nodes = len(node_names_by_type[endpoint_type])
    hcr_wide_values = torch.zeros((len(samples_df), n_endpoint_nodes), dtype=torch.float32)
    if hcr_wide_df is not None:
        if "patient_id" not in hcr_wide_df.columns:
            raise ValueError("hcr_wide_df musi zawierac kolumne 'patient_id'.")
        hcr_wide_indexed = hcr_wide_df.set_index("patient_id")
        used_endpoints = [c for c in hcr_wide_indexed.columns if c in node_names_by_type[endpoint_type]]
        if used_endpoints:
            aligned = hcr_wide_indexed.loc[samples_df["patient_id"], used_endpoints]
            aligned_values = torch.tensor(aligned.to_numpy(dtype=np.float32))
            local_positions = torch.tensor(
                [global_to_local[e][1] for e in used_endpoints], dtype=torch.long
            )
            hcr_wide_values[:, local_positions] = aligned_values
            print(f"[build_patient_dag_heterodata] HCR wide dolaczone dla endpointow: {used_endpoints}")

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
            # --- (G) kanal is_observed: 1 tylko dla kolumn realnie widocznych w tym reżimie ---
            observed_mask = torch.zeros(n_nodes, dtype=torch.float)
            if ntype in values_by_type:
                realized[col_local_idx_by_type[ntype]] = values_by_type[ntype][i]
                observed_mask[col_local_idx_by_type[ntype]] = 1.0

            static_feats = static_feats_by_type[ntype]  # [n_nodes, static_dim]
            x = torch.cat(
                [realized.unsqueeze(1), observed_mask.unsqueeze(1), static_feats], dim=1
            )
            data[ntype].x = x
            data[ntype].num_nodes = n_nodes
            # --- (E) indeks lokalny do embeddingu tozsamosci wezla w modelu ---
            data[ntype].node_idx = node_local_idx_by_type[ntype]

        # Etykiety i maska tylko na wezlach typu clinical_endpoint.
        y = torch.full((n_endpoint_nodes,), float("nan"), dtype=torch.float)
        y_mask = torch.zeros(n_endpoint_nodes, dtype=torch.bool)
        y[endpoint_local_idx] = y_values[i]
        y_mask[endpoint_local_idx] = True
        data[endpoint_type].y = y
        data[endpoint_type].y_mask = y_mask
        # HCR "wide" 
        data[endpoint_type].hcr_wide = hcr_wide_values[i]

        # Wspolna topologia krawedzi + edge_attr (waga + metadane audytu + one-hot
        for key, eidx in edge_index_dict.items():
            data[key].edge_index = eidx
            data[key].edge_attr = edge_attr_dict[key]

        data.patient_id = int(patient_ids[i])
        graphs.append(data)

    return graphs


# ---------------------------------------------------------------------------
# 3. Podzial train/valid/test
# ---------------------------------------------------------------------------

def load_split_map(split_path: Path) -> Dict[int, str]:
    split_df = pd.read_csv(split_path)
    return dict(zip(split_df["patient_id"].astype(int), split_df["split"]))


def attach_splits(graphs: List[HeteroData], split_path: Path) -> Dict[str, List[HeteroData]]:
    split_map = load_split_map(split_path)

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

def main(cfg, scenario: str = "clean"):
    topology = load_shared_hetero_topology(cfg)

    observability_regime = str(getattr(cfg.data.dataset, "observability_regime", "full"))

    samples_path = Path(cfg.data.dataset.root_dir) / f"synthetic_pharmacotherapy_v3_samples_{scenario}.csv"
    samples_df = pd.read_csv(samples_path, low_memory=False)

    # --- (K) statystyki normalizacyjne liczone WYLACZNIE na train, przed budowa grafow ---
    split_map = load_split_map(topology["splits_file"])
    train_patient_ids = {pid for pid, split in split_map.items() if split == "train"}
    norm_stats = compute_norm_stats(
        samples_df,
        topology["node_names_by_type"],
        train_patient_ids,
        exclude_cols=ID_COLS + META_COLS,
    )

    graphs = build_patient_hetero_graphs(
        samples_df,
        topology,
        observability_regime=observability_regime,
        norm_stats=norm_stats,
    )
    # --- (J) naprawione odwolanie do splits_file (wczesniej NameError) ---
    buckets = attach_splits(graphs, topology["splits_file"])

    loaders = {
        split: DataLoader(items, batch_size=cfg.training.batch_size, shuffle=(split == "train"))
        for split, items in buckets.items() if items
    }

    print(f"Scenario: {scenario}  |  observability_regime: {observability_regime}  |  layer_encoding: {topology['layer_encoding']} ({len(topology['layer_cols'])} kolumn)")
    print(f"Total patient hetero-graphs: {len(graphs)}")
    for split, items in buckets.items():
        print(f"  {split}: {len(items)} graphs")
    print(f"Node types: {topology['node_types']}")
    print(f"Relation types: {len(topology['edge_index_dict'])}  (edge_attr_dim={topology['edge_attr_dim']})")
    print(f"Excluded (latent/unobserved) nodes: {sorted(topology['excluded_nodes'])}")
    for ntype, names in topology["node_names_by_type"].items():
        visible = ntype in REGIME_VISIBLE_TYPES[observability_regime] and ntype != ENDPOINT_NODE_TYPE
        print(f"  {ntype}: {len(names)} nodes  (visible_as_feature={visible})")

    return loaders, topology


if __name__ == "__main__":
    main()
