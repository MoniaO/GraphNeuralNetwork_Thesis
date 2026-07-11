from __future__ import annotations

import random
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from data.load_split_benchmark_data import load_split_benchmark_heterodata
from models.gnn_lp import SimpleHeteroGNN
from evaluation.syntetic_evaluator import SynEvaluator


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg: DictConfig, train_data) -> SimpleHeteroGNN:
    metadata = train_data.metadata()
    in_dims = {
        node_type: int(train_data[node_type].x.size(-1))
        for node_type in train_data.node_types
    }
    return SimpleHeteroGNN(cfg=cfg, metadata=metadata, in_dims=in_dims)


def build_run_name(cfg: DictConfig) -> str:
    if "wandb" in cfg and getattr(cfg.wandb, "run_name", None):
        return str(cfg.wandb.run_name)
    return (
        f"{cfg.model.name}_{cfg.data.name}_{cfg.data.target}"
        f"_hd{cfg.model.hidden_dim}_L{cfg.model.num_layers}"
        f"_lr{cfg.training.lr}_seed{cfg.training.seed}"
    )


def train_epoch(model, data, optimizer, criterion, device: torch.device) -> float:
    model.train()
    data = data.to(device)

    optimizer.zero_grad()
    logits = model(data)
    labels = data[("patient", "has_adr", "variable")].edge_label.float()
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    return float(loss.item())


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    set_seed(int(cfg.training.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, valid_data, test_data = load_split_benchmark_heterodata(cfg)

    model = build_model(cfg, train_data).to(device)

    optimizer_name = str(getattr(cfg.training, "optimizer", "adam")).lower()
    lr = float(cfg.training.lr)
    weight_decay = float(getattr(cfg.training, "weight_decay", 0.0))

    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    pos_weight = float(getattr(cfg.training, "pos_weight", 1.0))
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
    )

    evaluator = SynEvaluator(cfg)

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
    best_state = None

    epochs = int(cfg.training.epochs)
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_data, optimizer, criterion, device)

        train_metrics = evaluator.evaluate(model, train_data, criterion, device)
        valid_metrics = evaluator.evaluate(model, valid_data, criterion, device)
        test_metrics = evaluator.evaluate(model, test_data, criterion, device)

        current_valid = valid_metrics.get(best_metric_name, float("nan"))
        if not np.isnan(current_valid) and current_valid > best_valid:
            best_valid = current_valid
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        log_dict = {
            "epoch": epoch,
            "train/optim_loss": train_loss,
            "train/loss": train_metrics["loss"],
            "train/auc": train_metrics["auc"],
            "train/auprc": train_metrics["auprc"],
            "train/brier": train_metrics["brier"],
            "valid/loss": valid_metrics["loss"],
            "valid/auc": valid_metrics["auc"],
            "valid/auprc": valid_metrics["auprc"],
            "valid/brier": valid_metrics["brier"],
            "test/loss": test_metrics["loss"],
            "test/auc": test_metrics["auc"],
            "test/auprc": test_metrics["auprc"],
            "test/brier": test_metrics["brier"],
            "best/valid_metric": best_valid,
            "best/epoch": best_epoch,
            "lr": optimizer.param_groups[0]["lr"],
        }

        if use_wandb:
            wandb.log(log_dict, step=epoch)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} | "
            f"valid AUC {valid_metrics['auc']:.4f} | "
            f"valid AUPRC {valid_metrics['auprc']:.4f} | "
            f"test AUC {test_metrics['auc']:.4f} | "
            f"test AUPRC {test_metrics['auprc']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    ckpt_path = output_dir / "best_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "best_epoch": best_epoch,
        "best_valid": best_valid,
    }, ckpt_path)

    if use_wandb:
        wandb.summary["best_epoch"] = best_epoch
        wandb.summary[f"best_valid_{best_metric_name}"] = best_valid
        wandb.summary["checkpoint_path"] = str(ckpt_path)
        wandb.finish()

    print(f"Saved checkpoint to: {ckpt_path}")
    print(f"Hydra run dir: {output_dir}")


if __name__ == "__main__":
    main()
