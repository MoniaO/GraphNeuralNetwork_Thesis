#!/usr/bin/env python3
"""Leakage-safe native heterogeneous GNN link prediction on v2.2.

Models:
  * Native Hetero GraphSAGE via HeteroConv (no ``to_hetero()``)
  * R-GCN over an explicitly flattened typed graph

Only positive training edges are available to message passing. Validation/test
positive edges and every negative candidate are withheld from graph structure.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch_geometric.nn import HeteroConv, RGCNConv, SAGEConv

from hetero_data_v2_2 import load_native_heterodata, validate_heterodata


PROJECT_ROOT = Path(__file__).resolve().parents[1]


SCENARIOS = [
    "clean",
    "hidden_confounder",
    "selection_bias",
    "no_overlap",
    "noisy_documentation",
    "multihospital",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "2. Data" / "dataset_v2_2",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "5. Results and reports"
        / "hetero_gnn_v2_2_outputs",
    )
    p.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--hidden-channels", type=int, default=32)
    p.add_argument("--negative-ratio", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260717)
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_candidates(data_dir: Path, negative_ratio: int, seed: int) -> pd.DataFrame:
    nodes = pd.read_csv(data_dir / "synthetic_pharmacotherapy_v2_2_nodes.csv")
    edges = pd.read_csv(data_dir / "synthetic_pharmacotherapy_v2_2_edges_audited.csv")
    latent = nodes["is_latent"].astype(str).str.lower().isin({"true", "1"})
    nodes = nodes.loc[~latent].copy()
    observed = set(nodes["node"])
    positives = edges.loc[
        edges["source"].isin(observed) & edges["target"].isin(observed)
    ].copy()
    node_type = nodes.set_index("node")["node_type"].to_dict()
    allowed_type_pairs = {
        (node_type[source], node_type[target])
        for source, target in positives[["source", "target"]].itertuples(
            index=False, name=None
        )
    }
    positive_pairs = set(
        positives[["source", "target"]].itertuples(index=False, name=None)
    )
    names = nodes["node"].tolist()
    negative_pool = [
        (source, target)
        for source in names
        for target in names
        if source != target
        and (node_type[source], node_type[target]) in allowed_type_pairs
        and (source, target) not in positive_pairs
    ]
    rng = np.random.default_rng(seed)
    count = min(len(negative_pool), negative_ratio * len(positives))
    sampled = rng.choice(len(negative_pool), size=count, replace=False)
    negatives = pd.DataFrame(
        [negative_pool[index] for index in sampled], columns=["source", "target"]
    )
    negatives["edge_id"] = ""
    negatives["edge_label"] = 0
    positives = positives[["source", "target", "edge_id"]].copy()
    positives["edge_label"] = 1
    candidates = pd.concat([positives, negatives], ignore_index=True)
    candidates.insert(
        0, "candidate_id", [f"v22_c{i:05d}" for i in range(len(candidates))]
    )
    candidates["source_type"] = candidates["source"].map(node_type)
    candidates["target_type"] = candidates["target"].map(node_type)
    return candidates


def add_split(candidates: pd.DataFrame, repeat: int, seed: int) -> pd.DataFrame:
    indices = np.arange(len(candidates))
    train_idx, held_idx = train_test_split(
        indices,
        test_size=0.30,
        random_state=seed + repeat,
        stratify=candidates["edge_label"],
    )
    val_idx, test_idx = train_test_split(
        held_idx,
        test_size=0.50,
        random_state=seed + 1000 + repeat,
        stratify=candidates.iloc[held_idx]["edge_label"],
    )
    result = candidates.copy()
    result["split"] = ""
    result.loc[train_idx, "split"] = "train"
    result.loc[val_idx, "split"] = "validation"
    result.loc[test_idx, "split"] = "test"
    result["edge_repeat"] = repeat
    return result


def global_layout(data) -> tuple[dict[str, int], dict[str, int]]:
    offsets: dict[str, int] = {}
    lookup: dict[str, int] = {}
    offset = 0
    for node_type in sorted(data.node_types):
        offsets[node_type] = offset
        for local_index, name in enumerate(data[node_type].node_name):
            lookup[name] = offset + local_index
        offset += data[node_type].num_nodes
    return offsets, lookup


def flatten_embeddings(data, embedding_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        [embedding_dict[node_type] for node_type in sorted(data.node_types)], dim=0
    )


class NativeHeteroSAGE(nn.Module):
    def __init__(self, data, in_channels: int, hidden_channels: int):
        super().__init__()
        relations = data.edge_types
        self.residual1 = nn.ModuleDict({
            node_type: nn.Linear(in_channels, hidden_channels)
            for node_type in data.node_types
        })
        self.residual2 = nn.ModuleDict({
            node_type: nn.Linear(hidden_channels, hidden_channels)
            for node_type in data.node_types
        })
        self.conv1 = HeteroConv(
            {
                relation: SAGEConv(
                    (in_channels, in_channels),
                    hidden_channels,
                )
                for relation in relations
            },
            aggr="sum",
        )
        self.conv2 = HeteroConv(
            {
                relation: SAGEConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels,
                )
                for relation in relations
            },
            aggr="sum",
        )

    def forward(self, data) -> torch.Tensor:
        messages1 = self.conv1(data.x_dict, data.edge_index_dict)
        hidden = {
            node_type: torch.relu(
                self.residual1[node_type](data[node_type].x)
                + messages1.get(node_type, 0.0)
            )
            for node_type in data.node_types
        }
        messages2 = self.conv2(hidden, data.edge_index_dict)
        output = {
            node_type: torch.relu(
                self.residual2[node_type](hidden[node_type])
                + messages2.get(node_type, 0.0)
            )
            for node_type in data.node_types
        }
        return flatten_embeddings(data, output)


class ExplicitRGCN(nn.Module):
    def __init__(self, data, in_channels: int, hidden_channels: int):
        super().__init__()
        self.node_types = sorted(data.node_types)
        self.offsets, _ = global_layout(data)
        self.relations = list(data.edge_types)
        self.conv1 = RGCNConv(
            in_channels,
            hidden_channels,
            num_relations=len(self.relations),
            num_bases=min(8, len(self.relations)),
        )
        self.conv2 = RGCNConv(
            hidden_channels,
            hidden_channels,
            num_relations=len(self.relations),
            num_bases=min(8, len(self.relations)),
        )

    def forward(self, data) -> torch.Tensor:
        x = torch.cat([data[node_type].x for node_type in self.node_types], dim=0)
        edge_parts, relation_parts = [], []
        for relation_id, key in enumerate(self.relations):
            edge_index = data[key].edge_index.clone()
            edge_index[0] += self.offsets[key[0]]
            edge_index[1] += self.offsets[key[2]]
            edge_parts.append(edge_index)
            relation_parts.append(
                torch.full(
                    (edge_index.shape[1],), relation_id, dtype=torch.long
                )
            )
        edge_index = torch.cat(edge_parts, dim=1)
        edge_type = torch.cat(relation_parts)
        hidden = torch.relu(self.conv1(x, edge_index, edge_type))
        return torch.relu(self.conv2(hidden, edge_index, edge_type))


class LinkDecoder(nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_channels, 1),
        )

    def forward(
        self, embeddings: torch.Tensor, source: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        source_z = embeddings[source]
        target_z = embeddings[target]
        pair = torch.cat(
            [
                source_z,
                target_z,
                source_z * target_z,
                torch.abs(source_z - target_z),
            ],
            dim=1,
        )
        return self.network(pair).squeeze(1)


def metrics(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    return (
        float(average_precision_score(labels, scores)),
        float(roc_auc_score(labels, scores)),
    )


def fit_one(
    architecture: str,
    loaded,
    split: pd.DataFrame,
    epochs: int,
    patience: int,
    hidden_channels: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    seed_everything(seed)
    data = loaded.data
    _, lookup = global_layout(data)
    source = torch.tensor(split["source"].map(lookup).to_numpy(), dtype=torch.long)
    target = torch.tensor(split["target"].map(lookup).to_numpy(), dtype=torch.long)
    labels = torch.tensor(split["edge_label"].to_numpy(), dtype=torch.float32)
    masks = {
        name: torch.tensor(split["split"].eq(name).to_numpy(), dtype=torch.bool)
        for name in ["train", "validation", "test"]
    }
    in_channels = data[data.node_types[0]].x.shape[1]
    if architecture == "hetero_sage":
        encoder = NativeHeteroSAGE(data, in_channels, hidden_channels)
    elif architecture == "rgcn":
        encoder = ExplicitRGCN(data, in_channels, hidden_channels)
    else:
        raise ValueError(architecture)
    decoder = LinkDecoder(hidden_channels)
    model = nn.ModuleDict({"encoder": encoder, "decoder": decoder})
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    positive = labels[masks["train"]].sum()
    negative = masks["train"].sum() - positive
    criterion = nn.BCEWithLogitsLoss(pos_weight=(negative / positive).clamp(min=1.0))
    best_state, best_ap, best_epoch, stale = None, -np.inf, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = decoder(encoder(data), source, target)
        loss = criterion(logits[masks["train"]], labels[masks["train"]])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(decoder(encoder(data), source, target))
        val_ap, _ = metrics(
            labels[masks["validation"]].numpy(),
            scores[masks["validation"]].numpy(),
        )
        if val_ap > best_ap + 1e-5:
            best_ap = val_ap
            best_epoch = epoch
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(decoder(encoder(data), source, target)).numpy()
    predictions = split.copy()
    predictions["score"] = scores
    row: dict[str, object] = {
        "architecture": architecture,
        "best_epoch": best_epoch,
        "validation_ap": best_ap,
    }
    for name in ["train", "validation", "test"]:
        mask = predictions["split"].eq(name).to_numpy()
        ap, auc = metrics(
            predictions.loc[mask, "edge_label"].to_numpy(),
            predictions.loc[mask, "score"].to_numpy(),
        )
        row[f"{name}_ap"] = ap
        row[f"{name}_roc_auc"] = auc
    return predictions, row


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = make_candidates(args.data_dir, args.negative_ratio, args.seed)
    candidates.to_csv(args.output_dir / "edge_candidates_v2_2.csv", index=False)
    summaries, all_predictions, split_tables = [], [], []
    for repeat in range(1, args.repeats + 1):
        split = add_split(candidates, repeat, args.seed)
        split_tables.append(split)
        positive_train_ids = set(
            split.loc[
                split["split"].eq("train") & split["edge_label"].eq(1), "edge_id"
            ]
        )
        for scenario in args.scenarios:
            loaded = load_native_heterodata(
                args.data_dir, scenario, train_edge_ids=positive_train_ids
            )
            diagnostics = validate_heterodata(loaded)
            for architecture in ["hetero_sage", "rgcn"]:
                predictions, summary = fit_one(
                    architecture,
                    loaded,
                    split,
                    args.epochs,
                    args.patience,
                    args.hidden_channels,
                    args.seed
                    + repeat * 100
                    + (0 if architecture == "hetero_sage" else 1),
                )
                predictions["scenario"] = scenario
                predictions["architecture"] = architecture
                predictions["edge_repeat"] = repeat
                all_predictions.append(predictions)
                summary.update({
                    "scenario": scenario,
                    "edge_repeat": repeat,
                    **diagnostics,
                })
                summaries.append(summary)
                print(json.dumps(summary))
    pd.concat(split_tables, ignore_index=True).to_csv(
        args.output_dir / "edge_splits_v2_2.csv", index=False
    )
    pd.concat(all_predictions, ignore_index=True).to_csv(
        args.output_dir / "hetero_gnn_predictions_v2_2.csv", index=False
    )
    pd.DataFrame(summaries).to_csv(
        args.output_dir / "hetero_gnn_summary_v2_2.csv", index=False
    )


if __name__ == "__main__":
    main()
