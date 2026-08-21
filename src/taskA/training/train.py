"""Task A training loop (link prediction on GSN v3).

What it does
------------
1. Loads the graph (`taskA.data.load_graph`) — G_train from positive train edges only.
2. If Stage C / FINAL: attaches S10 (`taskA.features.attach`).
3. Builds LinkPredictor (HGT + Fusion88).
4. Early-stops on valid AUPRC; test is computed once at the end (sealed).

What you may change
-------------------
`training.epochs`, `patience`, `lr`, `seed`, `device`, `wandb.enabled`.
FINAL freezes these in `experiments.final_14_08.FROZEN`.

What not to touch for FINAL
---------------------------
selection_metric=auprc, sealed test, pos_weight from train, candidate_seed=20260722.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from taskA.data.load_graph import load_recon_heterodata
from taskA.evaluation.metrics import SynEvaluator
from taskA.models.link_predictor import HeteroReconGNN
from taskA.features.attach import (
    fit_and_attach_stage_c_stats,
    stage_c_enabled,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(cfg: DictConfig) -> torch.device:
    requested = str(getattr(cfg.training, "device", "auto")).strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("training.device=cuda but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("training.device=mps but MPS is unavailable")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown training.device={requested!r} (auto|cpu|cuda|mps)")


class ReconEvaluator(SynEvaluator):
    """Task A adapter for top-level candidate-edge labels."""

    def __init__(self, cfg, node_to_idx=None):
        super().__init__(cfg, node_to_idx=node_to_idx)

    @torch.no_grad()
    def _forward_probs(self, model, data, criterion, device):
        model.eval()
        data = data.to(device)
        logits = model(data).view(-1)
        labels = data.edge_label.float().to(device).view(-1)
        probs = torch.sigmoid(logits)
        loss = criterion(logits, labels)
        return (
            data,
            labels.detach().cpu().numpy(),
            probs.detach().cpu().numpy(),
            float(loss.item()),
        )


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "parameter_count_total": int(total),
        "parameter_count_trainable": int(trainable),
    }


def build_model(cfg: DictConfig, train_data) -> HeteroReconGNN:
    in_channels = int(train_data[train_data.node_types[0]].x.size(-1))
    hidden_channels = int(
        getattr(cfg.model, "hidden_channels", None)
        or getattr(cfg.model, "hidden_dim", 32)
    )
    return HeteroReconGNN(
        cfg=cfg,
        data=train_data,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
    )


def build_run_name(cfg: DictConfig) -> str:
    if "wandb" in cfg and getattr(cfg.wandb, "run_name", None):
        return str(cfg.wandb.run_name)

    scenario = str(cfg.data.dataset.scenario)
    model_name = str(getattr(cfg.model, "conv_type", cfg.model.name))
    num_layers = int(cfg.model.num_layers)
    training_seed = int(cfg.training.seed)
    experiment = getattr(cfg, "experiment", None)
    wave = str(getattr(experiment, "wave", "FINAL") if experiment is not None else "FINAL")
    encoder = str(
        getattr(getattr(cfg.model, "decoder", None), "stat_pair_encoder", "mlp")
    )
    return (
        f"{wave}"
        f"__{scenario}"
        f"__{model_name}"
        f"__L{num_layers}"
        f"__{encoder}"
        f"__seed{training_seed}"
    )


def train_epoch(model, data, optimizer, criterion, device: torch.device) -> float:
    model.train()
    data = data.to(device)
    labels = data.edge_label.float().to(device).view(-1)
    optimizer.zero_grad(set_to_none=True)
    logits = model(data).view(-1)
    loss = criterion(logits, labels)
    decoder = getattr(model, "decoder", None)
    if decoder is not None and hasattr(decoder, "pair_encoder_regularization_loss"):
        reg = decoder.pair_encoder_regularization_loss()
        if torch.is_tensor(reg) and float(reg.detach()) != 0.0:
            loss = loss + reg.to(device=loss.device)
    loss.backward()
    clip = getattr(model, "grad_clip", None)
    if clip is not None and float(clip) > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
    optimizer.step()
    return float(loss.item())


def prefix_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            if value.numel() != 1:
                continue
            value = value.detach().cpu().item()
        if isinstance(value, (int, float, np.integer, np.floating)):
            out[f"{prefix}/{key}"] = float(value)
    return out


def flatten_oversmoothing(metrics: dict[str, Any], prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if not str(key).startswith("oversmoothing/"):
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            out[f"{prefix}/{key}"] = float(value)
    return out


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))

    seed = int(cfg.training.seed)
    set_seed(seed)
    device = resolve_device(cfg)
    print(f"\nUsing device: {device}")

    train_data, valid_data, test_data, node_to_idx = load_recon_heterodata(cfg)

    if stage_c_enabled(cfg):
        fit_and_attach_stage_c_stats(
            cfg,
            train_data,
            valid_data,
            test_data,
            device="cpu",
        )

    fingerprint = getattr(train_data, "candidate_fingerprint", None)
    candidate_seed = getattr(train_data, "candidate_seed", None)
    feature_ablation_profile = str(
        getattr(cfg.data, "feature_ablation_profile", "empirical")
    ).strip().lower()
    feature_ablation_seed = int(
        getattr(cfg.data, "feature_ablation_seed", 20260722)
    )
    effective_feature_fingerprint = str(
        getattr(train_data, "feature_fingerprint", "missing")
    )
    graph_fp = str(getattr(train_data, "graph_fingerprint", "missing"))
    experiment_name = str(getattr(cfg.experiment, "wave", "unknown"))
    print(
        f"\nCANDIDATE FREEZE"
        f"\n  candidate_seed:       {candidate_seed}"
        f"\n  candidate_fingerprint:{fingerprint}"
        f"\n  training.seed:        {seed}"
        f"\n  scenario:             {cfg.data.dataset.scenario}"
        f"\n  node_feature_profile: {getattr(cfg.data, 'node_feature_profile', None)}"
    )
    print(
        "\nEFFECTIVE INPUT FEATURES"
        f"\n  feature_ablation_profile: {feature_ablation_profile}"
        f"\n  feature_ablation_seed:    {feature_ablation_seed}"
        f"\n  feature_fingerprint:      {effective_feature_fingerprint}"
        f"\n  graph_fingerprint:        {graph_fp}"
        f"\n  experiment_name:          {experiment_name}"
    )

    model = build_model(cfg, train_data).to(device)
    model.grad_clip = float(getattr(cfg.training, "grad_clip", 0.0))

    with torch.no_grad():
        _ = model(train_data.to(device))

    param_counts = count_parameters(model)
    print(
        "\nPARAMETER COUNT"
        f"\n  total:     {param_counts['parameter_count_total']:,}"
        f"\n  trainable: {param_counts['parameter_count_trainable']:,}"
    )

    optimizer_name = str(getattr(cfg.training, "optimizer", "adam")).lower()
    lr = float(cfg.training.lr)
    weight_decay = float(getattr(cfg.training, "weight_decay", 0.0))
    opt_cls = torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.Adam
    optimizer = opt_cls(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_labels = train_data.edge_label.float()
    n_pos = float(train_labels.sum().clamp(min=1.0))
    n_neg = float((train_labels.numel() - train_labels.sum()).clamp(min=1.0))
    pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    eval_criterion = torch.nn.BCEWithLogitsLoss()

    evaluator = ReconEvaluator(cfg, node_to_idx=node_to_idx)

    use_wandb = bool(getattr(cfg.wandb, "enabled", True)) if "wandb" in cfg else False
    if use_wandb:
        wandb_tags = list(getattr(cfg.wandb, "tags", []) or [])
        wandb.init(
            project=str(cfg.wandb.project),
            entity=getattr(cfg.wandb, "entity", None),
            name=build_run_name(cfg),
            group=getattr(cfg.wandb, "group", None),
            job_type=getattr(cfg.wandb, "job_type", "training"),
            tags=wandb_tags,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb.summary["candidate_seed"] = candidate_seed
        wandb.summary["candidate_fingerprint"] = fingerprint
        wandb.summary["training_seed"] = seed
        wandb.summary["experiment_name"] = experiment_name
        wandb.summary["graph_fingerprint"] = graph_fp
        wandb.summary["feature_ablation_profile"] = feature_ablation_profile
        wandb.summary["feature_ablation_seed"] = feature_ablation_seed
        wandb.summary["feature_fingerprint"] = effective_feature_fingerprint
        wandb.summary["parameter_count_total"] = param_counts["parameter_count_total"]
        wandb.summary["parameter_count_trainable"] = param_counts[
            "parameter_count_trainable"
        ]
        wandb.summary["encoder_name"] = str(
            getattr(cfg.model, "encoder_name", getattr(cfg.model, "name", ""))
        )
        wandb.summary["hidden_channels"] = int(
            getattr(cfg.model, "hidden_channels", getattr(cfg.model, "hidden_dim", -1))
        )
        if hasattr(cfg.model, "heads"):
            wandb.summary["heads"] = int(cfg.model.heads)

    selection_metric = str(getattr(cfg.training, "selection_metric", "auprc")).strip().lower()
    if selection_metric.startswith("valid/"):
        selection_metric = selection_metric[len("valid/") :]

    patience = int(getattr(cfg.training, "early_stopping_patience", 20))
    min_delta = float(getattr(cfg.training, "early_stopping_min_delta", 0.0))
    epochs = int(cfg.training.epochs)

    best_valid = -float("inf")
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    last_epoch = 0
    diagnostic_threshold = 0.5

    for epoch in range(1, epochs + 1):
        last_epoch = epoch
        train_loss = train_epoch(model, train_data, optimizer, criterion, device)

        train_metrics = evaluator.evaluate(
            model, train_data, eval_criterion, device, threshold=diagnostic_threshold
        )
        valid_metrics = evaluator.evaluate(
            model, valid_data, eval_criterion, device, threshold=diagnostic_threshold
        )

        current_valid = float(valid_metrics.get(selection_metric, float("nan")))
        if np.isfinite(current_valid) and current_valid > best_valid + min_delta:
            best_valid = current_valid
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        log_dict: dict[str, float] = {
            "epoch": float(epoch),
            "train/optimization_loss": float(train_loss),
            "classification_threshold_diagnostic": diagnostic_threshold,
            "best/valid_metric": float(best_valid),
            "best/epoch": float(best_epoch),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        log_dict.update(prefix_metrics("train", train_metrics))
        log_dict.update(prefix_metrics("valid", valid_metrics))
        log_dict.update(flatten_oversmoothing(valid_metrics, "valid"))

        if use_wandb:
            wandb.log(log_dict, step=epoch)

        print(
            f"Epoch {epoch:03d} | optim {train_loss:.4f} | "
            f"train AUPRC {float(train_metrics.get('auprc', float('nan'))):.4f} | "
            f"valid AUPRC {float(valid_metrics.get('auprc', float('nan'))):.4f} | "
            f"valid AUC {float(valid_metrics.get('auc', float('nan'))):.4f} | "
            f"best {best_epoch:03d}"
        )

        if patience > 0 and epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch}: "
                f"no improvement in valid/{selection_metric} for {patience} epochs."
            )
            if use_wandb:
                wandb.summary["early_stopped"] = True
                wandb.summary["early_stopped_epoch"] = epoch
            break

    if best_epoch < 0:
        raise RuntimeError(
            f"No valid checkpoint selected (metric={selection_metric!r} never finite)."
        )

    model.load_state_dict(best_state)
    model.to(device)

    final_threshold = float(evaluator.select_threshold(model, valid_data, device))
    print(f"\nFinal threshold (valid): {final_threshold:.6f}")

    final_train = evaluator.evaluate(
        model, train_data, eval_criterion, device, threshold=final_threshold
    )
    final_valid = evaluator.evaluate(
        model, valid_data, eval_criterion, device, threshold=final_threshold
    )
    final_test = evaluator.evaluate(
        model, test_data, eval_criterion, device, threshold=final_threshold
    )

    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "best_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "cfg": OmegaConf.to_container(cfg, resolve=True),
            "best_epoch": best_epoch,
            "best_valid_metric_name": selection_metric,
            "best_valid_metric": best_valid,
            "classification_threshold": final_threshold,
            "candidate_seed": candidate_seed,
            "candidate_fingerprint": fingerprint,
            "final_train_metrics": final_train,
            "final_valid_metrics": final_valid,
            "final_test_metrics": final_test,
        },
        ckpt_path,
    )

    final_log: dict[str, float] = {
        "final/classification_threshold": final_threshold,
        "final/best_epoch": float(best_epoch),
        "final/best_valid_metric": float(best_valid),
    }
    final_log.update(prefix_metrics("final_train", final_train))
    final_log.update(prefix_metrics("final_valid", final_valid))
    final_log.update(prefix_metrics("final_test", final_test))
    final_log.update(flatten_oversmoothing(final_valid, "final_valid"))
    final_log.update(flatten_oversmoothing(final_test, "final_test"))

    if use_wandb:
        wandb.log(final_log, step=last_epoch + 1)
        wandb.summary["best_epoch"] = best_epoch
        wandb.summary["best_valid_metric_name"] = selection_metric
        wandb.summary["best_valid_metric"] = best_valid
        wandb.summary["best_valid_AUPRC"] = best_valid
        wandb.summary["classification_threshold"] = final_threshold
        wandb.summary["checkpoint_path"] = str(ckpt_path)
        for key, value in final_test.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                wandb.summary[f"test_{key}"] = float(value)
        for key, value in final_valid.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                wandb.summary[f"valid_{key}"] = float(value)
        wandb.finish()

    print("\nFINAL RESULTS")
    print(f"Best epoch: {best_epoch}")
    print(f"Best valid/{selection_metric}: {best_valid:.6f}")
    print(f"Final threshold: {final_threshold:.6f}")
    print(f"Final valid AUPRC: {float(final_valid.get('auprc', float('nan'))):.6f}")
    print(f"Final test AUPRC:  {float(final_test.get('auprc', float('nan'))):.6f}")
    print(f"Final test AUC:    {float(final_test.get('auc', float('nan'))):.6f}")
    print(f"Saved checkpoint:  {ckpt_path}")
    print(f"Hydra run dir:     {output_dir}")


if __name__ == "__main__":
    main()
