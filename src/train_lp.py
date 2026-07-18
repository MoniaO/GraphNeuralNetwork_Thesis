from __future__ import annotations

import copy
import random
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf, ListConfig

from data.load_split_benchmark_data import load_split_benchmark_heterodata
from evaluation.syntetic_evaluator import SynEvaluator
from models.gnn_lp import SimpleHeteroGNN
from training.class_weights import compute_pos_weights
from models.linear import LinearHeteroLP


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_model(cfg: DictConfig, train_data):
    metadata = train_data.metadata()
    in_dims = {
        node_type: int(train_data[node_type].x.size(-1))
        for node_type in train_data.node_types
    }
    model_name = str(cfg.model.name).lower()

    if model_name in {"linear", "linear_lp", "linear_hetero_lp"}:
        return LinearHeteroLP(cfg=cfg, metadata=metadata, in_dims=in_dims)

    if model_name in {"gnn_lp", "gnn_lp_attr", "heterognn", "simple_hetero_gnn"}:
        return SimpleHeteroGNN(cfg=cfg, metadata=metadata, in_dims=in_dims)

    raise ValueError(f"Unknown model.name='{cfg.model.name}'")

def build_run_name(cfg: DictConfig) -> str:
    if "wandb" in cfg and getattr(cfg.wandb, "run_name", None):
        return str(cfg.wandb.run_name)
    
    target = cfg.data.target
    if isinstance(target, (list, tuple, ListConfig)):
        targets_str = "multitarget" if len(target) > 5 else "-".join(target)
    else:
        targets_str = str(target)

    scenario = getattr(cfg.data.dataset, "scenario", None)
    scenario_str = str(scenario) if scenario is not None else "default"


    return (
        f"{cfg.meta.owner_initials}_{cfg.model.name}_{cfg.data.name}"
        f"_{targets_str}_{cfg.model.conv_type}_{scenario_str}"
        f"_ep{cfg.training.epochs}_layer{cfg.model.num_layers}"
        f"_lr{cfg.training.lr}"
        f"_bs{cfg.training.batch_size}"
    )


def get_labels(data) -> torch.Tensor:
    return data[("patient", "has_adr", "variable")].edge_label.float()


def train_epoch(model, data, optimizer, criterion, device: torch.device) -> float:
    model.train()
    data = data.to(device)
    labels = get_labels(data)

    optimizer.zero_grad()
    logits = model(data)
    loss = criterion(logits, labels)
    loss.backward()

    grad_clip = getattr(model, "grad_clip", None)
    if grad_clip is not None and grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()
    return float(loss.item())


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    set_seed(int(cfg.training.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, valid_data, test_data, node_to_idx = load_split_benchmark_heterodata(cfg)

    model = build_model(cfg, train_data).to(device)

    optimizer_name = str(getattr(cfg.training, "optimizer", "adam")).lower()
    lr = float(cfg.training.lr)
    weight_decay = float(getattr(cfg.training, "weight_decay", 0.0))

    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    #for just singlelabel
    #pos_weight = float(getattr(cfg.training, "pos_weight", 1.0))

    pos_weight_map = compute_pos_weights(train_data, node_to_idx)

    target_idx_per_row = train_data[("patient", "has_adr", "variable")].edge_label_target_idx
    pos_weight_tensor = torch.tensor(
    [pos_weight_map[int(t)] for t in target_idx_per_row],
    dtype=torch.float32,
    device=device)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    eval_criterion = torch.nn.BCEWithLogitsLoss()

    evaluator = SynEvaluator(cfg, node_to_idx=node_to_idx)
    

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

    epochs = int(cfg.training.epochs)
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_data, optimizer, criterion, device)

        train_metrics = evaluator.evaluate(model, train_data, eval_criterion, device)
        valid_metrics = evaluator.evaluate(model, valid_data, eval_criterion, device)
        test_metrics = evaluator.evaluate(model, test_data, eval_criterion, device)

        print({k: v for k, v in train_metrics.items() if k.startswith("oversmoothing/")})

        current_valid = valid_metrics.get(best_metric_name, float("nan"))
        if not np.isnan(current_valid) and current_valid > best_valid:
            best_valid = current_valid
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())


        log_dict = {
            "epoch": epoch,
            "train/optim_loss": train_loss,

            "train/loss": train_metrics["loss"],
            "train/auc": train_metrics["auc"],
            "train/auprc": train_metrics["auprc"],
            "train/brier": train_metrics["brier"],
            "train/f1": train_metrics["f1"],
            "train/precision": train_metrics["precision"],
            "train/recall": train_metrics["recall"],
            "train/auprc_baseline": train_metrics["auprc_baseline"],
            "train/auprc_lift": train_metrics["auprc_lift"],

            "valid/loss": valid_metrics["loss"],
            "valid/auc": valid_metrics["auc"],
            "valid/auprc": valid_metrics["auprc"],
            "valid/brier": valid_metrics["brier"],
            "valid/f1": valid_metrics["f1"],
            "valid/precision": valid_metrics["precision"],
            "valid/recall": valid_metrics["recall"],
            "valid/auprc_baseline": valid_metrics["auprc_baseline"],
            "valid/auprc_lift": valid_metrics["auprc_lift"],

            "test/loss": test_metrics["loss"],
            "test/auc": test_metrics["auc"],
            "test/auprc": test_metrics["auprc"],
            "test/brier": test_metrics["brier"],
            "test/f1": test_metrics["f1"],
            "test/precision": test_metrics["precision"],
            "test/recall": test_metrics["recall"],
            "test/auprc_baseline": test_metrics["auprc_baseline"],
            "test/auprc_lift": test_metrics["auprc_lift"],

            "best/valid_metric": best_valid,
            "best/epoch": best_epoch,
            "lr": optimizer.param_groups[0]["lr"],
        }

        for split_name, metrics_dict in [
            ("train", train_metrics),
            ("valid", valid_metrics),
            ("test", test_metrics),
            ]:
            for key, value in metrics_dict.items():
                if key.startswith(("auc_", "auprc_", "oversmoothing/")):
                    log_dict[f"{split_name}/{key}"] = value

        if use_wandb:
            wandb.log(log_dict, step=epoch)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} | "
            f"train AUC {train_metrics['auc']:.4f} | "
            f"train AUPRC {train_metrics['auprc']:.4f} | "
            f"valid loss {valid_metrics['loss']:.4f} | "
            f"test loss {test_metrics['loss']:.4f} | "
            f"valid AUC {valid_metrics['auc']:.4f} | "
            f"valid AUPRC {valid_metrics['auprc']:.4f} | "
            f"test AUC {test_metrics['auc']:.4f} | "
            f"test AUPRC {test_metrics['auprc']:.4f}"
        )

    model.load_state_dict(best_state)

    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    ckpt_path = output_dir / "best_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "cfg": OmegaConf.to_container(cfg, resolve=True),
            "best_epoch": best_epoch,
            "best_valid": best_valid,
        },
        ckpt_path,
    )

    if use_wandb:
        wandb.summary["best_epoch"] = best_epoch
        wandb.summary[f"best_valid_AUPRC"] = best_valid
        wandb.summary["checkpoint_path"] = str(ckpt_path)
        wandb.finish()

    print(f"Saved checkpoint to: {ckpt_path}")
    print(f"Hydra run dir: {output_dir}")


if __name__ == "__main__":
    main()