"""
Training entrypoint for Task A: structural link prediction / audited-graph
reconstruction (Hetero GraphSAGE / R-GCN), following the same Hydra + wandb
+ SynEvaluator pattern as train_lp.py.

Task A predicts `edge_label` (does this (source, target) pair exist in the
audited causal graph)
""" 

from __future__ import annotations

import copy
import random
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from data.PreprocessingTaskA.load_hetero_recon_data import load_recon_heterodata
from evaluation.syntetic_evaluator import SynEvaluator
from models.TaskA.hetero_gnn import HeteroReconGNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ReconEvaluator(SynEvaluator):
    """SynEvaluator is hardcoded to the ('patient','has_adr','variable') edge
    type used in train_lp.py (Task B), in two places:
      1. _forward_probs() reads labels from that edge type.
      2. evaluate()'s per-endpoint branch reads edge_label_target_idx from
         that same edge type, to break metrics down by endpoint.

    Task A has a single, uniform target (edge_label: does this node pair
    form an edge), not multiple endpoints -- there is no per-endpoint
    breakdown to compute. So per_endpoint is forced off here, regardless of
    cfg.training.eval_per_endpoint (which stays True for Task B runs).
    All ranking/calibration/oversmoothing logic is otherwise reused unchanged.
    """

    def __init__(self, cfg, node_to_idx=None):
        super().__init__(cfg, node_to_idx=node_to_idx)
        self.per_endpoint = False

    @torch.no_grad()
    def _forward_probs(self, model, data, criterion, device):
        model.eval()
        data = data.to(device)
        logits = model(data)
        labels = data.edge_label.float()
        probs = torch.sigmoid(logits)
        loss = criterion(logits, labels).item()
        return data, labels.detach().cpu().numpy(), probs.detach().cpu().numpy(), float(loss)


def build_model(cfg: DictConfig, train_data) -> HeteroReconGNN:
    in_channels = int(train_data[train_data.node_types[0]].x.size(-1))
    hidden_channels = int(getattr(cfg.model, "hidden_channels", 32))
    return HeteroReconGNN(cfg=cfg, data=train_data, in_channels=in_channels, hidden_channels=hidden_channels)


def build_run_name(cfg: DictConfig) -> str:
    if "wandb" in cfg and getattr(cfg.wandb, "run_name", None):
        return str(cfg.wandb.run_name)
    scenario = str(cfg.data.dataset.scenario)
    return (
        f"{cfg.meta.owner_initials}_TaskA_{cfg.model.name}_"
        f"_{scenario}_ep{cfg.training.epochs}"
        f"_lr{cfg.training.lr}_layers{cfg.model.num_layers}_hidden{cfg.model.hidden_channels}"
    )


def train_epoch(model, data, optimizer, criterion, device: torch.device) -> float:
    model.train()
    data = data.to(device)
    labels = data.edge_label.float()

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
    train_data, valid_data, test_data, node_to_idx = load_recon_heterodata(cfg)

    model = build_model(cfg, train_data).to(device)

    optimizer_name = str(getattr(cfg.training, "optimizer", "adam")).lower()
    lr = float(cfg.training.lr)
    weight_decay = float(getattr(cfg.training, "weight_decay", 0.0))
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scalar pos_weight from the train-split class balance (edge_label is a
    # single binary target here, unlike the multi-endpoint case in train_lp.py,
    # so compute_pos_weights()'s per-endpoint map is not applicable).
    train_labels = train_data.edge_label
    n_pos = train_labels.sum().clamp(min=1.0)
    n_neg = (train_labels.numel() - n_pos).clamp(min=1.0)
    pos_weight_tensor = (n_neg / n_pos).to(device)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    eval_criterion = torch.nn.BCEWithLogitsLoss()

    evaluator = ReconEvaluator(cfg, node_to_idx=node_to_idx)
    threshold = evaluator.select_threshold(model, valid_data, device)

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
        train_loss = train_epoch(model, train_data, optimizer, criterion, device)

        threshold = evaluator.select_threshold(model, valid_data, device)
        train_metrics = evaluator.evaluate(model, train_data, eval_criterion, device, threshold=threshold)
        valid_metrics = evaluator.evaluate(model, valid_data, eval_criterion, device, threshold=threshold)
        test_metrics = evaluator.evaluate(model, test_data, eval_criterion, device, threshold=threshold)

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
            "train/loss": train_metrics["loss"], "train/auc": train_metrics["auc"],
            "train/auprc": train_metrics["auprc"], "train/brier": train_metrics["brier"],
            "train/f1": train_metrics["f1"], "train/precision": train_metrics["precision"],
            "train/recall": train_metrics["recall"], "train/auprc_baseline": train_metrics["auprc_baseline"],
            "train/auprc_lift": train_metrics["auprc_lift"],
            "valid/loss": valid_metrics["loss"], "valid/auc": valid_metrics["auc"],
            "valid/auprc": valid_metrics["auprc"], "valid/brier": valid_metrics["brier"],
            "valid/f1": valid_metrics["f1"], "valid/precision": valid_metrics["precision"],
            "valid/recall": valid_metrics["recall"], "valid/auprc_baseline": valid_metrics["auprc_baseline"],
            "valid/auprc_lift": valid_metrics["auprc_lift"],
            "test/loss": test_metrics["loss"], "test/auc": test_metrics["auc"],
            "test/auprc": test_metrics["auprc"], "test/brier": test_metrics["brier"],
            "test/f1": test_metrics["f1"], "test/precision": test_metrics["precision"],
            "test/recall": test_metrics["recall"], "test/auprc_baseline": test_metrics["auprc_baseline"],
            "test/auprc_lift": test_metrics["auprc_lift"],
            "best/valid_metric": best_valid, "best/epoch": best_epoch,
            "lr": optimizer.param_groups[0]["lr"],
        }
        for split_name, metrics_dict in [("train", train_metrics), ("valid", valid_metrics), ("test", test_metrics)]:
            for key, value in metrics_dict.items():
                if key.startswith(("auc_", "auprc_", "oversmoothing/")):
                    log_dict[f"{split_name}/{key}"] = value

        if use_wandb:
            wandb.log(log_dict, step=epoch)

        print(
            f"Epoch {epoch:03d} | train loss {train_loss:.4f} | "
            f"valid AUPRC {valid_metrics['auprc']:.4f} | test AUPRC {test_metrics['auprc']:.4f} | "
            f"valid AUC {valid_metrics['auc']:.4f} | test AUC {test_metrics['auc']:.4f}"
        )

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}: no improvement in valid/{best_metric_name} for {patience} epochs.")
            if use_wandb:
                wandb.summary["early_stopped"] = True
                wandb.summary["early_stopped_epoch"] = epoch
            break

    model.load_state_dict(best_state)

    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    ckpt_path = output_dir / "best_model.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "cfg": OmegaConf.to_container(cfg, resolve=True),
         "best_epoch": best_epoch, "best_valid": best_valid},
        ckpt_path,
    )

    if use_wandb:
        wandb.summary["best_epoch"] = best_epoch
        wandb.summary["best_valid_AUPRC"] = best_valid
        wandb.summary["checkpoint_path"] = str(ckpt_path)
        wandb.finish()

    print(f"Saved checkpoint to: {ckpt_path}")
    print(f"Hydra run dir: {output_dir}")


if __name__ == "__main__":
    main()
