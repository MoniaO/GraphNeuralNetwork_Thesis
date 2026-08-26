from __future__ import annotations

import copy
import random
from pathlib import Path
import os

import hydra
import numpy as np
import pandas as pd
import torch
import wandb
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf, ListConfig
from torch_geometric.loader import DataLoader

from data.PreprocessingTaskB.hcr_wide_features import (
    get_binary_direct_parents,
    get_direct_parents_by_node_type,
    compute_hcr_pair_evidence,
    compute_hcr_pair_evidence_40d, 
    compute_hcr_pair_evidence_mixed, 
    get_direct_parents_by_type,
    compute_hcr_pair_evidence_by_node_type, 
    compute_hcr_wide_scores, 
    keep_observed,
    PARENT_NODE_TYPES
)

from data.PreprocessingTaskB.hcr_triple_features import (
    get_triples_by_endpoint,
    compute_hcr_triple_evidence
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

from data.PreprocessingTaskB.hcr_keep_visible import keep_visible, REGIME_VISIBLE_TYPES

from evaluation.syntetic_evaluator_node import SynEvaluatorNode
from models.TaskB.gnn_node import (
    SimplePatientDAGNodeClassifier,
    TargetedPatientDAGNodeClassifier,
    get_targeted_labels,
    DEFAULT_TARGET_ENDPOINTS,
)
from models.TaskB.gnn_rgcn_node import RGCNPatientDAGNodeClassifier
from training.losses import ClassBalanceStats, compute_class_balance_stats, build_criterion


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
    visible = REGIME_VISIBLE_TYPES[observability_regime]
    excluded = set(topology["excluded_nodes"])

    # w build_loaders(), po wczytaniu topology i samples_df:
    raw_targets = getattr(cfg.data, "target", DEFAULT_TARGET_ENDPOINTS)

    if isinstance(raw_targets, str):
        target_endpoints = [raw_targets]
    else:
        target_endpoints = list(raw_targets)

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
    #parents_by_endpoint = get_binary_direct_parents(
    #    nodes_df, edges_df, raw_targets, excluded_nodes
    #)


    split_map = load_split_map(topology["splits_file"])
    # Statystyki normalizacyjne  na train
    train_patient_ids = {pid for pid, split in split_map.items() if split == "train"}

    use_hcr = bool(getattr(cfg.model, "use_hcr_wide", False))
    hcr_wide_df = None
    pair_evidence = None
    hcr_pair_evidence_by_type = None

    if use_hcr:
        hcr_parent_scope = str(getattr(cfg.model, "hcr_parent_scope", "binary")).lower()
        hcr_evidence_dim = int(getattr(cfg.model, "hcr_evidence_dim", 9))
        hcr_wide_mode = str(getattr(cfg.model, "hcr_wide_mode", "nonlinear")).lower()
        hcr_parent_scope = str(getattr(cfg.model, "hcr_parent_scope", "binary")).lower()

        parents_by_endpoint = get_binary_direct_parents(nodes_df, edges_df, target_endpoints, excluded_nodes)
        #parents_by_endpoint = keep_observed(parents_by_endpoint, samples_df)
        parents_by_endpoint = keep_visible(parents_by_endpoint, samples_df, nodes_df, visible, excluded)
        hcr_wide_df = compute_hcr_wide_scores(samples_df, parents_by_endpoint, train_patient_ids)


        if hcr_wide_mode == "linear":
            pass  # nic dodatkowego - hcr_wide_df jest i tak liczone bezwarunkowo wyzej

        elif hcr_wide_mode == "nonlinear":
            # hcr_parent_scope ma znaczenie - jako PODWYBOR wewnatrz "nonlinear"
            if hcr_parent_scope == "binary":
                parents_by_endpoint = keep_observed(get_binary_direct_parents(nodes_df, edges_df, target_endpoints, excluded_nodes), samples_df)
                if hcr_evidence_dim == 10:
                    pair_evidence = compute_hcr_pair_evidence(samples_df, parents_by_endpoint, train_patient_ids)
                elif hcr_evidence_dim == 42:
                    pair_evidence = compute_hcr_pair_evidence_40d(samples_df, parents_by_endpoint, train_patient_ids, edges=edges_df)
                else:
                    raise ValueError(f"hcr_evidence_dim={hcr_evidence_dim} nieobslugiwane dla scope='binary' (9 lub 41).")
            elif hcr_parent_scope == "all":
                if hcr_evidence_dim != 42:
                    raise ValueError("hcr_parent_scope='all' wymaga hcr_evidence_dim=42.")
                parents_by_endpoint_type = keep_observed(get_direct_parents_by_type(nodes_df, edges_df, target_endpoints, excluded_nodes), samples_df)
                pair_evidence = compute_hcr_pair_evidence_mixed(samples_df, parents_by_endpoint_type, train_patient_ids, edges=edges_df)
            else:
                raise ValueError(f"hcr_parent_scope={hcr_parent_scope!r} nieznane (binary lub all).")

        elif hcr_wide_mode == "typed_concat":
            if hcr_evidence_dim != 42:
                raise ValueError("hcr_wide_mode='typed_concat' wymaga hcr_evidence_dim=41.")
            parents_by_endpoint_by_type = keep_observed(get_direct_parents_by_node_type(nodes_df, edges_df, target_endpoints, excluded_nodes), samples_df)
            hcr_pair_evidence_by_type = compute_hcr_pair_evidence_by_node_type(
                    samples_df, parents_by_endpoint_by_type, train_patient_ids, edges=edges_df
                )
            pair_evidence = None
            if bool(getattr(cfg.model, "hcr_use_triples", False)):
                triples = get_triples_by_endpoint(nodes_df, edges_df, target_endpoints, excluded_nodes, set(samples_df.columns), 
                                                  samples_df=samples_df, train_patient_ids=train_patient_ids)
                print(f"[HCR triple] trojek per endpoint: {[len(v) for v in triples.values()]}")
                pair_evidence = compute_hcr_triple_evidence(samples_df, triples, train_patient_ids)
                topology["triple_stats"] = {
                    "n_triples": pair_evidence["n_triples"],
                    "n_estimable": pair_evidence["n_estimable"],
                }

        elif hcr_wide_mode == "triple":
            observed = set(samples_df.columns)
            triples = get_triples_by_endpoint(
                nodes_df, edges_df, target_endpoints, excluded_nodes, observed, samples_df=samples_df, train_patient_ids=train_patient_ids)
            pair_evidence = compute_hcr_triple_evidence(samples_df, triples, train_patient_ids)
            print(f"[HCR triple] trojek per endpoint: "
                f"{ {ep: len(v) for ep, v in triples.items()} }")
        else:
                raise ValueError(f"cfg.model.hcr_wide_mode={hcr_wide_mode!r} nieznane (linear/nonlinear/typed_concat).")

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
        hcr_pair_evidence=pair_evidence, 
        hcr_pair_evidence_by_type=hcr_pair_evidence_by_type
    )

    buckets = attach_splits(graphs, topology["splits_file"])

    batch_size = int(cfg.training.batch_size)
    loaders = {
        split: DataLoader(items, batch_size=batch_size, shuffle=(split == "train"))
        for split, items in buckets.items() if items
    }
    return loaders, topology, graphs


# ---------------------------------------------------------------------------
# Model
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
    regime = str(getattr(cfg.data.dataset, "observability_regime", "full"))

    use_residual = bool(getattr(cfg.model, "use_residual", True))
    residual_str = "res" if use_residual else "nores"
    use_hcr = bool(getattr(cfg.model, "use_hcr_wide", True))
    use_hcr_str = "HCR" if use_hcr else "NoHCR"
    mode = str(getattr(cfg.model, "hcr_wide_mode", "linear")).lower() if use_hcr else ""
    detailed = use_hcr and mode not in ("", "linear")
 
    use_hcr_mode   = f"_{mode}" if use_hcr else ""
    use_hcr_parent = f"_{getattr(cfg.model, 'hcr_parent_scope', '')}" if detailed else ""
    use_hcr_dim    = f"_{getattr(cfg.model, 'hcr_evidence_dim', '')}" if detailed else ""
    use_hcr_hid_dim    = f"_{getattr(cfg.model, 'hcr_hidden_dim', '')}" if detailed else ""

    return (
        f"TaskB_{cfg.model.hide_direct_parents}EGoptTR{cfg.model.hcr_use_triples}{use_hcr_str}{use_hcr_mode}{use_hcr_parent}{use_hcr_dim}{use_hcr_hid_dim}_{cfg.model.name}_s{cfg.training.seed}"
        f"_{targets_str}_{cfg.model.conv_type}_{cfg.data.name}_{scenario_str}_{regime}_{residual_str}"
        f"_ep{cfg.training.epochs}_L{cfg.model.num_layers}"
        f"_lr{cfg.training.lr}_hid{cfg.model.hidden_dim}"
        f"_bs{cfg.training.batch_size}_aggr_{cfg.model.aggr}_nb_{getattr(cfg.model, 'num_bases', '')}_dropedge_{getattr(cfg.model, 'drop_edge', 0.0)}_jk_{getattr(cfg.model, 'jk_mode', '')}"
    )



# batch training per epoch

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

    gate_names = ("hcr_type_gate", "hcr_endpoint_weight", "hcr_wide_weight")
    gate_params, other_params = [], []
    for name, p in model.named_parameters():
        (gate_params if any(g in name for g in gate_names) else other_params).append(p)

    groups = [{"params": other_params, "weight_decay": weight_decay}]

    if gate_params:
        groups.append({"params": gate_params, "weight_decay": 0.0})
        print(f"[optim] {len(gate_params)} parametrow bramy HCR bez weight_decay")


    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(groups, lr=lr, weight_decay=weight_decay)

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

    # evaluation criteration - BCEWithLogitsLoss
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

            valid_metrics = evaluator.evaluate(
                model, loaders["validation"], eval_criterion, device, select_threshold=True
            )
            threshold = valid_metrics["classification_threshold"]

            scheduler.step(valid_metrics[best_metric_name])

            do_full_eval = (epoch % 10 == 0) or (epoch == epochs)
            train_metrics, test_metrics = {}, {}
            if do_full_eval:
                #threshold from valid set
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

            #DODANE JEŚLI MAMY ZOSTAĆ PRZY HCR PER ENDPOINT
            if getattr(model, "hcr_endpoint_weight", None) is not None:
                weights = dict(zip(model.target_endpoint_names,
                                   model.hcr_endpoint_weight.detach().cpu().tolist()))
                log_dict.update({f"hcr_endpoint_weight/{ep}": w for ep, w in weights.items()})

            if getattr(model, "hcr_type_gate", None) is not None:
                # macierz [n_targets, n_types] -> po jednej serii na pare
                gate = model.hcr_type_gate.detach().cpu()
                block_names = list(PARENT_NODE_TYPES)
                if gate.size(1) == len(block_names) + 1:
                    block_names.append("triple")
                for j, ep_name in enumerate(model.target_endpoint_names):
                    for i, nt in enumerate(block_names):
                        log_dict[f"hcr_gate/{ep_name}/{nt}"] = float(gate[j, i])
                log_dict["hcr_gate/max_abs"] = float(gate.abs().max())
                log_dict["hcr_gate/mean_abs"] = float(gate.abs().mean())

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

        # po zakonczeniu treningu
            if getattr(model, "hcr_type_gate", None) is not None:
                gate = model.hcr_type_gate.detach().cpu()

                # Etykiety kolumn MUSZA isc za rzeczywista szerokoscia macierzy:
                # przy wlaczonych trojkach jest ich piec, nie cztery.
                block_names = list(PARENT_NODE_TYPES)
                if gate.size(1) == len(block_names) + 1:
                    block_names.append("triple")

                # HCR gate matrix
                lim = max(float(gate.abs().max()), 1e-6)
                fig, ax = plt.subplots(figsize=(6, 8))
                im = ax.imshow(gate.numpy(), aspect="auto", cmap="RdBu_r",
                               vmin=-lim, vmax=lim)
                ax.set_xticks(range(len(block_names)))
                ax.set_xticklabels(block_names, rotation=45, ha="right")
                ax.set_yticks(range(len(model.target_endpoint_names)))
                ax.set_yticklabels(model.target_endpoint_names)
                ax.set_title(f"max |gate| = {float(gate.abs().max()):.4f}")
                fig.colorbar(im)
                fig.tight_layout()
                wandb.summary["hcr_gate_matrix"] = wandb.Image(fig)
                plt.close(fig)

                wandb.summary["hcr_gate/max_abs"] = float(gate.abs().max())
                wandb.summary["hcr_gate/mean_abs"] = float(gate.abs().mean())

            # tryb nonlinear ma wektor, nie macierz - wystarcza same liczby
            if getattr(model, "hcr_endpoint_weight", None) is not None:
                w = model.hcr_endpoint_weight.detach().cpu()
                for name, v in zip(model.target_endpoint_names, w.tolist()):
                    wandb.summary[f"hcr_endpoint_weight/{name}"] = v
                wandb.summary["hcr_endpoint_weight/max_abs"] = float(w.abs().max())

            ts = topology.get("triple_stats")
            if ts:
                wandb.summary["hcr_triple/n_triples"] = ts["n_triples"]
                wandb.summary["hcr_triple/n_estimable"] = ts["n_estimable"]
                wandb.summary["hcr_triple/frac_estimable"] = (
                    ts["n_estimable"] / max(ts["n_triples"], 1))

    finally:
        if use_wandb:
            wandb.finish() if use_wandb else None


if __name__ == "__main__":
    main()
