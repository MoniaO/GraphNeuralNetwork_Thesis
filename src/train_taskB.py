from __future__ import annotations

import copy
import random
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf, ListConfig
from torch_geometric.loader import DataLoader

from data.PreprocessingTaskB.build_patient_dag_heterodata import (
    load_shared_hetero_topology,
    build_patient_hetero_graphs,
    attach_splits
)
from evaluation.syntetic_evaluator_node import SynEvaluatorNode
from models.TaskB.gnn_node import (
    TargetedPatientDAGNodeClassifier,
    get_targeted_labels,
    DEFAULT_TARGET_ENDPOINTS,
)
from models.TaskB.gnn_node import SimplePatientDAGNodeClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Budowa danych: loadery train/valid/test na bazie grafow-pacjentow
# ---------------------------------------------------------------------------

def build_loaders(cfg: DictConfig):
    dataset_cfg = cfg.data.dataset
    root = Path(dataset_cfg.root_dir)

    topology = load_shared_hetero_topology(cfg)

    scenario = str(getattr(dataset_cfg, "scenario", "clean"))
    samples_file = root / f"synthetic_pharmacotherapy_v3_samples_{scenario}.csv"

    import pandas as pd
    samples_df = pd.read_csv(samples_file, low_memory=False)

    graphs = build_patient_hetero_graphs(samples_df, topology)

    split_file = root / "splits" / str(getattr(dataset_cfg, "splits_file", "patient_splits_v3.csv"))
    buckets = attach_splits(graphs, split_file)

    batch_size = int(cfg.training.batch_size)
    loaders = {
        split: DataLoader(items, batch_size=batch_size, shuffle=(split == "train"))
        for split, items in buckets.items() if items
    }
    return loaders, topology, graphs


# ---------------------------------------------------------------------------
# Budowa modelu
# ---------------------------------------------------------------------------

def build_model(cfg: DictConfig, topology: dict, sample_graph):
    node_types = sample_graph.node_types
    edge_types = sample_graph.edge_types
    metadata = (node_types, edge_types)

    in_dims = {
        node_type: int(sample_graph[node_type].x.size(-1))
        for node_type in node_types
    }

    raw_targets = getattr(cfg.data, "target", DEFAULT_TARGET_ENDPOINTS)
    if isinstance(raw_targets, str):
        target_endpoints = [raw_targets]
    else:
        target_endpoints = list(raw_targets)

    model_name = str(cfg.model.name).lower()

    if model_name in {"gnn_node_clf_simple", "baseline_node_clf"}:
        return SimplePatientDAGNodeClassifier(
            cfg=cfg, metadata=metadata, in_dims=in_dims,
            target_node_type="clinical_endpoint",
        )

    if model_name in {"gnn_node"}:
        return TargetedPatientDAGNodeClassifier(
            cfg=cfg, metadata=metadata, in_dims=in_dims,
            node_names_by_type=topology["node_names_by_type"],
            target_node_type="clinical_endpoint",
            target_endpoint_names=target_endpoints,
        )

    raise ValueError(f"Unknown model.name='{cfg.model.name}'")


def build_run_name(cfg: DictConfig) -> str:
    if "wandb" in cfg and getattr(cfg.wandb, "run_name", None):
        return str(cfg.wandb.run_name)

    target_endpoints = list(getattr(cfg.data, "target_endpoints", DEFAULT_TARGET_ENDPOINTS))
    targets_str = "multitarget" if len(target_endpoints) > 5 else "-".join(target_endpoints)

    scenario = getattr(cfg.data.dataset, "scenario", None)
    scenario_str = str(scenario) if scenario is not None else "default"

    return (
        f"{cfg.meta.owner_initials}_{cfg.model.name}_{cfg.data.name}"
        f"_{targets_str}_{cfg.model.conv_type}_{scenario_str}"
        f"_ep{cfg.training.epochs}_layer{cfg.model.num_layers}"
        f"_lr{cfg.training.lr}"
        f"_bs{cfg.training.batch_size}"
    )


# ---------------------------------------------------------------------------
# pos_weight per endpoint, liczony z loadera treningowego (nie z jednego grafu)
# ---------------------------------------------------------------------------

def compute_pos_weights_from_loader(train_loader: DataLoader, target_endpoint_names, target_local_idx, device: torch.device) -> torch.Tensor:
    """Zlicza pozytywne/negatywne etykiety per endpoint po WSZYSTKICH
    grafach w loaderze treningowym, zwraca tensor pos_weight [n_targets]
    do uzycia w BCEWithLogitsLoss (per-kolumna, broadcastowane na batch)."""
    n_targets = len(target_endpoint_names)
    positives = torch.zeros(n_targets)
    totals = torch.zeros(n_targets)

    for batch in train_loader:
        batch = batch.to(device)
        labels = get_targeted_labels(batch, target_local_idx, target_node_type="clinical_endpoint")
        labels = labels.view(-1, n_targets)
        positives += labels.sum(dim=0).cpu()
        totals += labels.size(0)

    negatives = totals - positives
    pos_weight = negatives / positives.clamp(min=1.0)
    return pos_weight


# ---------------------------------------------------------------------------
# Trening jednej epoki: iteracja po batchach, nie jeden forward na cala populacje
# ---------------------------------------------------------------------------

def train_epoch(model, loader: DataLoader, optimizer, criterion, device: torch.device) -> float:
    model.train()
    total_loss, n_batches = 0.0, 0
    n_targets = len(model.target_endpoint_names)

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        labels = get_targeted_labels(batch, model.target_local_idx, target_node_type=model.target_node_type)

        optimizer.zero_grad()
        logits = model(batch)
        
        logits = logits.view(-1, n_targets)
        labels = labels.view(-1, n_targets)
        
        loss = criterion(logits, labels)
        loss.backward()

        grad_clip = getattr(model, "grad_clip", None)
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    set_seed(int(cfg.training.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, topology, graphs = build_loaders(cfg)
    sample_graph = graphs[0]

    model = build_model(cfg, topology, sample_graph).to(device)

    optimizer_name = str(getattr(cfg.training, "optimizer", "adam")).lower()
    lr = float(cfg.training.lr)
    weight_decay = float(getattr(cfg.training, "weight_decay", 0.0))

    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    pos_weight = compute_pos_weights_from_loader(
        loaders["train"], model.target_endpoint_names, model.target_local_idx, device
    ).to(device)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    eval_criterion = torch.nn.BCEWithLogitsLoss()

    evaluator = SynEvaluatorNode(cfg, target_endpoint_names=model.target_endpoint_names)
    threshold = evaluator.select_threshold(model, loaders["validation"], device)

    use_wandb = bool(getattr(cfg.wandb, "enabled", True)) if "wandb" in cfg else False
    if use_wandb:
        wandb.init(
            project=cfg.wandb.project,
            entity=getattr(cfg.wandb, "entity", None),
            name=build_run_name(cfg),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    best_metric_name = str(getattr(cfg.training, "selection_metric", "auprc"))
    best_valid = -float("inf")
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())

    patience = int(getattr(cfg.training, "early_stopping_patience", 20))
    min_delta = float(getattr(cfg.training, "early_stopping_min_delta", 0.0))
    epochs_without_improvement = 0

    epochs = int(cfg.training.epochs)
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, loaders["train"], optimizer, criterion, device)
        valid_metrics = evaluator.evaluate(model, loaders["validation"], eval_criterion, device, threshold=threshold)

        do_full_eval = (epoch % 10 == 0) or (epoch == epochs)
        train_metrics, test_metrics = {}, {}
        if do_full_eval:
            train_metrics = evaluator.evaluate(model, loaders["train"], eval_criterion, device, threshold=threshold)
            test_metrics = evaluator.evaluate(model, loaders["test"], eval_criterion, device, threshold=threshold)
            print({k: v for k, v in train_metrics.items() if k.startswith("oversmoothing/")})

        current_valid = valid_metrics.get(best_metric_name, float("nan"))
        if not np.isnan(current_valid) and current_valid > best_valid + min_delta:
            best_valid = current_valid
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        log_dict = {
            "epoch": epoch,
            "train/optim_loss": train_loss,
            "classification_threshold": threshold,

            "valid/loss": valid_metrics["loss"],
            "valid/auc": valid_metrics["auc"],
            "valid/auprc": valid_metrics["auprc"],
            "valid/brier": valid_metrics["brier"],
            "valid/f1": valid_metrics["f1"],
            "valid/precision": valid_metrics["precision"],
            "valid/recall": valid_metrics["recall"],
            "valid/auprc_baseline": valid_metrics["auprc_baseline"],
            "valid/auprc_lift": valid_metrics["auprc_lift"],

            "best/valid_metric": best_valid,
            "best/epoch": best_epoch,
            "lr": optimizer.param_groups[0]["lr"],
        }

        for key, value in valid_metrics.items():
            if key.startswith(tuple(model.target_endpoint_names)) or key.startswith("oversmoothing/"):
                log_dict[f"valid/{key}"] = value

        if do_full_eval:
            log_dict.update({
                "train/loss": train_metrics["loss"],
                "train/auc": train_metrics["auc"],
                "train/auprc": train_metrics["auprc"],
                "train/brier": train_metrics["brier"],
                "train/f1": train_metrics["f1"],
                "train/precision": train_metrics["precision"],
                "train/recall": train_metrics["recall"],
                "train/auprc_baseline": train_metrics["auprc_baseline"],
                "train/auprc_lift": train_metrics["auprc_lift"],

                "test/loss": test_metrics["loss"],
                "test/auc": test_metrics["auc"],
                "test/auprc": test_metrics["auprc"],
                "test/brier": test_metrics["brier"],
                "test/f1": test_metrics["f1"],
                "test/precision": test_metrics["precision"],
                "test/recall": test_metrics["recall"],
                "test/auprc_baseline": test_metrics["auprc_baseline"],
                "test/auprc_lift": test_metrics["auprc_lift"],
            })
            for split_name, metrics_dict in [("train", train_metrics), ("test", test_metrics)]:
                for key, value in metrics_dict.items():
                    if key.startswith(tuple(model.target_endpoint_names)) or key.startswith("oversmoothing/"):
                        log_dict[f"{split_name}/{key}"] = value

        if use_wandb:
            wandb.log(log_dict, step=epoch)

        if do_full_eval:
            print(
                f"Epoch {epoch:03d} | train loss {train_loss:.4f} | "
                f"train AUC {train_metrics['auc']:.4f} | train AUPRC {train_metrics['auprc']:.4f} | "
                f"valid loss {valid_metrics['loss']:.4f} | test loss {test_metrics['loss']:.4f} | "
                f"valid AUC {valid_metrics['auc']:.4f} | valid AUPRC {valid_metrics['auprc']:.4f} | "
                f"test AUC {test_metrics['auc']:.4f} | test AUPRC {test_metrics['auprc']:.4f}"
            )
        else:
            print(
                f"Epoch {epoch:03d} | train loss {train_loss:.4f} | "
                f"valid loss {valid_metrics['loss']:.4f} | "
                f"valid AUC {valid_metrics['auc']:.4f} | valid AUPRC {valid_metrics['auprc']:.4f}"
            )

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch}: no improvement in valid/{best_metric_name} for {patience} epochs.")
            if use_wandb:
                wandb.summary["early_stopped"] = True
                wandb.summary["early_stopped_epoch"] = epoch
            break

    model.load_state_dict(best_state)


if __name__ == "__main__":
    main()
