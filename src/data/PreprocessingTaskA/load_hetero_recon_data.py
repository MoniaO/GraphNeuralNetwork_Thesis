"""
Data loader for Task A (structural link prediction on the audited v2.2 graph),
built on top of the colleague's hetero_data_v2_2.load_native_heterodata().

Produces three HeteroData objects (train / valid / test) that all share the
SAME message-passing graph (built only from training-split positive edges,
per the leakage-safety rule), but carry DIFFERENT candidate pairs to score:

    data.link_source_idx : LongTensor [num_candidates_in_this_split]
    data.link_target_idx : LongTensor [num_candidates_in_this_split]
    data.edge_label       : FloatTensor [num_candidates_in_this_split]

Indices are *global* flat indices across node types (see global_layout in
hetero_recon_gnn.py), consistent with how HeteroReconGNN.forward() indexes
into its flattened embedding tensor.

This mirrors load_split_benchmark_heterodata(cfg) in data/load_split_benchmark_data.py:
same signature style (reads from cfg), same return shape (train, valid, test, node_to_idx).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from data.PreprocessingTaskA.hetero_data_v2_2 import load_native_heterodata


def _global_layout(data) -> dict[str, int]:
    offsets: dict[str, int] = {}
    offset = 0
    for node_type in sorted(data.node_types):
        offsets[node_type] = offset
        offset += data[node_type].num_nodes
    return offsets


def make_candidates(data_dir: Path, negative_ratio: int, seed: int) -> pd.DataFrame:
    """Unchanged logic from the colleague's make_candidates(), kept local so
    this loader has no import-time dependency on her training script."""
    nodes = pd.read_csv(data_dir / "synthetic_pharmacotherapy_v2_2_nodes.csv")
    edges = pd.read_csv(data_dir / "synthetic_pharmacotherapy_v2_2_edges_audited.csv")
    latent = nodes["is_latent"].astype(str).str.lower().isin({"true", "1"})
    nodes = nodes.loc[~latent].copy()
    observed = set(nodes["node"])
    positives = edges.loc[edges["source"].isin(observed) & edges["target"].isin(observed)].copy()
    node_type = nodes.set_index("node")["node_type"].to_dict()
    allowed_type_pairs = {
        (node_type[s], node_type[t]) for s, t in positives[["source", "target"]].itertuples(index=False, name=None)
    }
    positive_pairs = set(positives[["source", "target"]].itertuples(index=False, name=None))
    names = nodes["node"].tolist()
    negative_pool = [
        (s, t)
        for s in names
        for t in names
        if s != t and (node_type[s], node_type[t]) in allowed_type_pairs and (s, t) not in positive_pairs
    ]
    rng = np.random.default_rng(seed)
    count = min(len(negative_pool), negative_ratio * len(positives))
    sampled = rng.choice(len(negative_pool), size=count, replace=False)
    negatives = pd.DataFrame([negative_pool[i] for i in sampled], columns=["source", "target"])
    negatives["edge_id"] = ""
    negatives["edge_label"] = 0
    positives = positives[["source", "target", "edge_id"]].copy()
    positives["edge_label"] = 1
    candidates = pd.concat([positives, negatives], ignore_index=True)
    candidates.insert(0, "candidate_id", [f"v22_c{i:05d}" for i in range(len(candidates))])
    return candidates


def add_split(candidates: pd.DataFrame, repeat: int, seed: int) -> pd.DataFrame:
    """Unchanged logic from the colleague's add_split()."""
    indices = np.arange(len(candidates))
    train_idx, held_idx = train_test_split(
        indices, test_size=0.30, random_state=seed + repeat, stratify=candidates["edge_label"]
    )
    val_idx, test_idx = train_test_split(
        held_idx, test_size=0.50, random_state=seed + 1000 + repeat,
        stratify=candidates.iloc[held_idx]["edge_label"],
    )
    result = candidates.copy()
    result["split"] = ""
    result.loc[train_idx, "split"] = "train"
    result.loc[val_idx, "split"] = "validation"
    result.loc[test_idx, "split"] = "test"
    result["edge_repeat"] = repeat
    return result


def _attach_candidates(data, split_df: pd.DataFrame, split_name: str, lookup: dict[str, int]):
    part = split_df.loc[split_df["split"].eq(split_name)]
    data.link_source_idx = torch.tensor(part["source"].map(lookup).to_numpy(), dtype=torch.long)
    data.link_target_idx = torch.tensor(part["target"].map(lookup).to_numpy(), dtype=torch.long)
    data.edge_label = torch.tensor(part["edge_label"].to_numpy(), dtype=torch.float32)
    return data


def _resolve_data_dir(cfg) -> Path:
    """cfg.data.dataset.root_dir is typically ${oc.env:PHARMA_DATA_ROOT}.
    Fails fast with a clear message if the env var / path isn't set up,
    instead of a bare OmegaConf/FileNotFoundError deeper in pandas.read_csv.
    """
    raw = cfg.data.dataset.get("root_dir", None)
    if raw is None or str(raw).strip() == "":
        raise ValueError(
            "cfg.data.dataset.root_dir is empty. If it uses "
            "${oc.env:PHARMA_DATA_ROOT}, set the environment variable before "
            "running, e.g.: export PHARMA_DATA_ROOT=/path/to/dataset_v2_2"
        )
    data_dir = Path(str(raw)).expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Resolved data_dir does not exist: {data_dir}\n"
            f"Check PHARMA_DATA_ROOT / cfg.data.dataset.root_dir."
        )
    nodes_file = data_dir / str(cfg.data.dataset.get("nodes_file", "synthetic_pharmacotherapy_v2_2_nodes.csv"))
    edges_file = data_dir / str(cfg.data.dataset.get("edges_file", "synthetic_pharmacotherapy_v2_2_edges_audited.csv"))
    for path in (nodes_file, edges_file):
        if not path.exists():
            raise FileNotFoundError(
                f"Expected file not found: {path}\n"
                f"Check cfg.data.dataset.root_dir and cfg.data.dataset.nodes_file/edges_file."
            )
    return data_dir


def load_recon_heterodata(cfg):
    """cfg is expected to expose (mirroring load_split_benchmark_heterodata style,
    matching the colleague's dataset_syn.yaml schema):

        cfg.data.dataset.root_dir     -> ${oc.env:PHARMA_DATA_ROOT}, path to the v2.2 dataset folder
        cfg.data.dataset.scenario     -> scenario token, e.g. "clean"
        cfg.data.negative_ratio       -> negatives sampled per positive edge
        cfg.data.edge_repeat          -> which repeated split to use (1..3)
        cfg.training.seed             -> base seed

    Returns (train_data, valid_data, test_data, node_to_idx), same shape as
    load_split_benchmark_heterodata(cfg).
    """
    data_dir = _resolve_data_dir(cfg)
    scenario = str(cfg.data.dataset.scenario)
    negative_ratio = int(getattr(cfg.data, "negative_ratio", 3))
    edge_repeat = int(getattr(cfg.data, "edge_repeat", 1))
    seed = int(cfg.training.seed)

    candidates = make_candidates(data_dir, negative_ratio, seed)
    split_df = add_split(candidates, edge_repeat, seed)

    positive_train_ids = set(
        split_df.loc[split_df["split"].eq("train") & split_df["edge_label"].eq(1), "edge_id"]
    )

    # Message-passing graph: built ONCE from train-only positive edges. The
    # SAME graph object (deep-copied per split) is reused for valid/test so
    # that encoder weights see identical structure; only candidate pairs
    # (link_source_idx/link_target_idx/edge_label) differ across splits.
    loaded = load_native_heterodata(data_dir, scenario, train_edge_ids=positive_train_ids)
    base_data = loaded.data
    offsets = _global_layout(base_data)
    lookup: dict[str, int] = {}
    for node_type, (nt, local_idx) in loaded.node_lookup.items():
        lookup[node_type] = offsets[nt] + local_idx

    import copy as _copy

    train_data = _attach_candidates(_copy.copy(base_data), split_df, "train", lookup)
    valid_data = _attach_candidates(_copy.copy(base_data), split_df, "validation", lookup)
    test_data = _attach_candidates(_copy.copy(base_data), split_df, "test", lookup)

    return train_data, valid_data, test_data, lookup
