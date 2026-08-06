from __future__ import annotations

import copy
import random
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import wandb
from omegaconf import DictConfig, OmegaConf, ListConfig
from torch_geometric.loader import DataLoader

from data.PreprocessingTaskB.hcr_wide_features import (
    get_binary_direct_parents,
    compute_hcr_pair_evidence,
    compute_hcr_pair_evidence_40d
)


from data.PreprocessingTaskB.build_patient_dag_heterodata_regime import (
    load_shared_hetero_topology,
    build_patient_hetero_graphs,
    attach_splits,
    compute_norm_stats,
    load_split_map,
    ID_COLS,
    META_COLS,
)
from evaluation.syntetic_evaluator_node import SynEvaluatorNode
from models.TaskB.gnn_node import (
    SimplePatientDAGNodeClassifier,
    TargetedPatientDAGNodeClassifier,
    get_targeted_labels,
    DEFAULT_TARGET_ENDPOINTS,
)
from models.TaskB.gnn_rgcn_node import RGCNPatientDAGNodeClassifier
from training.losses import ClassBalanceStats, compute_class_balance_stats, build_criterion

from data.PreprocessingTaskB.hcr_wide_features import (
    get_binary_direct_parents, compute_hcr_wide_scores, compute_hcr_pair_evidence 
)


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
    samples_df = pd.read_csv(samples_file, low_memory=False)

    # Reżim obserwowalności 
    observability_regime = str(getattr(dataset_cfg, "observability_regime", "full"))

    # w build_loaders(), po wczytaniu topology i samples_df:
    raw_targets = getattr(cfg.data, "target", DEFAULT_TARGET_ENDPOINTS)

    dataset_cfg = cfg.data.dataset
    root = Path(dataset_cfg.root_dir)

    nodes_file = root / str(getattr(dataset_cfg, "nodes_file", "synthetic_pharmacotherapy_v3_nodes.csv"))
    edges_file = root / str(
        getattr(dataset_cfg, "edges_file", "synthetic_pharmacotherapy_v3_edges_audited.csv")
    )

    #hcr temp - to reconstruct
    excluded_nodes = topology["excluded_nodes"]
    nodes_df = pd.read_csv(nodes_file)
    edges_df = pd.read_csv(edges_file)
    parents_by_endpoint = get_binary_direct_parents(
        nodes_df, edges_df, raw_targets, excluded_nodes
    )




    split_map = load_split_map(topology["splits_file"])
    # Statystyki normalizacyjne  na train
    train_patient_ids = {pid for pid, split in split_map.items() if split == "train"}

    hcr_evidence_dim = int(getattr(cfg.model, "hcr_evidence_dim", 9))
    if hcr_evidence_dim == 9:
        pair_evidence = compute_hcr_pair_evidence(samples_df, parents_by_endpoint, train_patient_ids)
    elif hcr_evidence_dim == 41:
        pair_evidence = compute_hcr_pair_evidence_40d(samples_df, parents_by_endpoint, train_patient_ids)
    else:
        raise ValueError(f"cfg.model.hcr_evidence_dim={hcr_evidence_dim} nieobslugiwane (9 lub 41).")

    hcr_wide_df = compute_hcr_wide_scores(samples_df, parents_by_endpoint, train_patient_ids)

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
        hcr_wide_df=hcr_wide_df,
        hcr_pair_evidence=pair_evidence
    )

    buckets = attach_splits(graphs, topology["splits_file"])

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

    edge_dim = int(topology["edge_attr_dim"])

    raw_targets = getattr(cfg.data, "target", DEFAULT_TARGET_ENDPOINTS)
    if isinstance(raw_targets, str):
        target_endpoints = [raw_targets]
    else:
        target_endpoints = list(raw_targets)

    model_name = str(cfg.model.name).lower()

    if model_name in {"gnn_node_clf_simple", "baseline_node_clf"}:
        return SimplePatientDAGNodeClassifier(
            cfg=cfg, metadata=metadata, in_dims=in_dims,
            node_names_by_type=topology["node_names_by_type"],
            target_node_type="clinical_endpoint",
            edge_dim=edge_dim,
        )

    if model_name in {"gnn_node"}:
        return TargetedPatientDAGNodeClassifier(
            cfg=cfg, metadata=metadata, in_dims=in_dims,
            node_names_by_type=topology["node_names_by_type"],
            target_node_type="clinical_endpoint",
            target_endpoint_names=target_endpoints,
            edge_dim=edge_dim,
        )

    if model_name in {"rgcn", "gnn_node_rgcn"}:
        return RGCNPatientDAGNodeClassifier(
            cfg=cfg, topology=topology, in_dims=in_dims,
            target_node_type="clinical_endpoint",
            target_endpoint_names=target_endpoints,
        )

    raise ValueError(f"Unknown model.name='{cfg.model.name}'")


def build_run_name(cfg: DictConfig) -> str:
    if "wandb" in cfg and getattr(cfg.wandb, "run_name", None):
        return str(cfg.wandb.run_name)

    raw_targets = getattr(cfg.data, "target", DEFAULT_TARGET_ENDPOINTS)
    if isinstance(raw_targets, str):
        target_endpoints = [raw_targets]
    else:
        target_endpoints = list(raw_targets)

    targets_str = "multitarget" if len(target_endpoints) > 5 else "-".join(target_endpoints)

    scenario = getattr(cfg.data.dataset, "scenario", None)
    scenario_str = str(scenario) if scenario is not None else "default"

    # observability_regime dolaczony do nazwy runu - inaczej rozne reżimy
    # ("full" / "mechanisms_latent" / "bedside") nadpisywalyby sie nawzajem
    # przy takiej samej reszcie konfiguracji w porownaniach W&B.
    regime = str(getattr(cfg.data.dataset, "observability_regime", "full"))

    use_residual = bool(getattr(cfg.model, "use_residual", True))
    residual_str = "res" if use_residual else "nores"

    return (
        f"{cfg.meta.owner_initials}_TaskB_HCRExtended2_exnoisy_{cfg.model.name}_s{cfg.training.seed}"
        f"_{targets_str}_{cfg.model.conv_type}_{scenario_str}_{regime}_{residual_str}"
        f"_ep{cfg.training.epochs}_L{cfg.model.num_layers}"
        f"_lr{cfg.training.lr}_hid{cfg.model.hidden_dim}"
        f"_bs{cfg.training.batch_size}_aggr_{cfg.model.aggr}_nb_{getattr(cfg.model, 'num_bases', '')}_dropedge_{getattr(cfg.model, 'drop_edge', 0.0)}_jk_{getattr(cfg.model, 'jk_mode', '')}"
    )



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

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-5)

    label_extractor = lambda batch: get_targeted_labels(
        batch, model.target_local_idx, target_node_type=model.target_node_type
    )
    stats = compute_class_balance_stats(
        loaders["train"], label_extractor, len(model.target_endpoint_names), device
    )
    criterion = build_criterion(cfg, stats, device)

    loss_type = str(getattr(cfg.training, "loss_type", "bce")).lower()
    print(f"loss_type={loss_type}")
    print("prevalence per endpoint:", dict(zip(cfg.data.target, stats.prevalence.tolist())))
    if loss_type == "bce":
        print("pos_weight per endpoint:", dict(zip(cfg.data.target, stats.pos_weight().tolist())))
    elif loss_type == "focal":
        print(f"focal_gamma={getattr(cfg.training, 'focal_gamma', 2.0)}")
        print("focal_alpha per endpoint:", dict(zip(cfg.data.target, stats.focal_alpha().tolist())))

    # eval_criterion zostaje CELOWO zwyklym, niewazonym BCE niezaleznie od
    # loss_type - to jest wspolna, porownywalna skala "loss" w logach W&B
    # miedzy roznymi ustawieniami (bce/focal/rozne gamma). Gdyby eval_criterion
    # tez byl focal loss, wartosci "valid/loss" miedzy runami o roznym gamma
    # nie bylyby ze soba porownywalne (inna skala liczbowa strat).
    eval_criterion = torch.nn.BCEWithLogitsLoss()

    noisy_endpoints = {"Serotonin_syndrome", "Rhabdomyolysis", "Lactic_acidosis"}
    metric_target = [e for e in model.target_endpoint_names if e not in noisy_endpoints]

    evaluator = SynEvaluatorNode(
        cfg,
        target_endpoint_names=model.target_endpoint_names,
        metric_endpoint_names=metric_target,
    )

    use_wandb = bool(getattr(cfg.wandb, "enabled", True)) if "wandb" in cfg else False
    regime = str(getattr(cfg.data.dataset, "observability_regime", "full"))
    use_residual = bool(getattr(cfg.model, "use_residual", True))
    targets = "multitarget" if len(cfg.data.target) > 5 else "-".join(cfg.data.target)

    loss_tag = loss_type if loss_type != "focal" else f"focal_g{getattr(cfg.training, 'focal_gamma', 2.0)}"

    wandb_tags = [
        "TaskB",
        f"conv={cfg.model.conv_type}",
        f"regime={regime}",
        f"layers={cfg.model.num_layers}",
        f"residual={'res' if use_residual else 'nores'}",
        f"scenario={getattr(cfg.data.dataset, 'scenario', 'clean')}",
        f"model={cfg.model.name}",
        f"targets={targets}",
        f"loss={loss_tag}",
    ]
    wandb_group = f"{targets}_{cfg.model.conv_type}_{regime}_L{cfg.model.num_layers}"

    if use_wandb:
        wandb.init(
            project=cfg.wandb.project,
            entity=getattr(cfg.wandb, "entity", None),
            name=build_run_name(cfg),
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=wandb_tags,
            group=wandb_group,
        )

    try:

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

            # Prog wyznaczany TUTAJ, jako czesc tego samego forward-passu co
            # valid_metrics (select_threshold=True) - NIE osobnym wywolaniem
            # evaluator.select_threshold(), ktore zrobiloby DRUGIE, niezalezne
            # przejscie po loaders["validation"]. Dzieki temu prog jest swiezy
            # co epoke przy DOKLADNIE takim samym koszcie obliczeniowym, jaki
            # bylby bez zadnego mechanizmu przeliczania progu w ogole.
            valid_metrics = evaluator.evaluate(
                model, loaders["validation"], eval_criterion, device, select_threshold=True
            )
            threshold = valid_metrics["classification_threshold"]

            scheduler.step(valid_metrics[best_metric_name])

            do_full_eval = (epoch % 10 == 0) or (epoch == epochs)
            train_metrics, test_metrics = {}, {}
            if do_full_eval:
                # train/test uzywaja progu WYZNACZONEGO NA WALIDACJI powyzej
                # (nie wlasnego) - prog zawsze powinien pochodzic z valid,
                # nigdy z danych, na ktorych jest raportowany wynik.
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

            if getattr(model, "hcr_wide_weight", None) is not None:
                weights = dict(zip(model.target_endpoint_names, model.hcr_wide_weight.detach().cpu().tolist()))
                log_dict.update({f"hcr_wide_weight/{ep}": w for ep, w in weights.items()})

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

        # --- Finalna ewaluacja na najlepszym checkpoincie ---
        # Tak samo jak w petli: prog wyznaczany w TYM SAMYM forward-passie co
        # final_valid_metrics (select_threshold=True), zamiast osobnym
        # wywolaniem select_threshold() - jeden przebieg po walidacji, nie dwa.
        final_valid_metrics = evaluator.evaluate(
            model, loaders["validation"], eval_criterion, device, select_threshold=True
        )
        final_threshold = final_valid_metrics["classification_threshold"]
        final_test_metrics = evaluator.evaluate(
            model, loaders["test"], eval_criterion, device, threshold=final_threshold
        )

        print(
            f"[FINAL best_epoch={best_epoch}] threshold={final_threshold:.4f} | "
            f"valid AUC {final_valid_metrics['auc']:.4f} | valid AUPRC {final_valid_metrics['auprc']:.4f} | "
            f"test AUC {final_test_metrics['auc']:.4f} | test AUPRC {final_test_metrics['auprc']:.4f}"
        )

        if use_wandb:
            wandb.summary["final/best_epoch"] = best_epoch
            wandb.summary["final/threshold"] = final_threshold
            for key, value in final_valid_metrics.items():
                wandb.summary[f"final/valid/{key}"] = value
            for key, value in final_test_metrics.items():
                wandb.summary[f"final/test/{key}"] = value

    finally:
        if use_wandb:
            wandb.finish() if use_wandb else None


if __name__ == "__main__":
    main()
