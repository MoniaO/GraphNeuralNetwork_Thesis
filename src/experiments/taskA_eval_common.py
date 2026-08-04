"""Shared helpers for Task A post-hoc / frozen evaluation scripts (E1–E3)."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
SRC_DIR = PROJECT_ROOT / "src"
CHECKPOINT_ROOT = PROJECT_ROOT / "outputs" / "taskA_E1_checkpoints"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.PreprocessingTaskA.load_hetero_recon_data import load_recon_heterodata
from models.TaskA.hetero_gnn import HeteroReconGNN
from train_taskA import (
    ReconEvaluator,
    build_model,
    resolve_device,
    set_seed,
    train_epoch,
)


SEEDS = [20260721, 20260722, 20260723, 20260724, 20260725]
FEATURE_ABLATION_SEED = 20260722
CANDIDATE_SEED = 20260722

MODEL_SPECS = (
    ("hetero_sage", "TaskA_hetero_sage", 2),
    ("rgcn", "TaskA_rgcn", 1),
)


def compose_task_a_cfg(
    *,
    hydra_model: str,
    num_layers: int,
    training_seed: int,
    feature_ablation_profile: str = "empirical",
    epochs: int = 100,
    relation_ablation_enabled: bool = False,
    removed_relations: list[str] | None = None,
) -> DictConfig:
    removed = list(removed_relations or [])
    removed_override = "[" + ",".join(removed) + "]"
    overrides = [
        f"model={hydra_model}",
        f"model.num_layers={num_layers}",
        "data.dataset.scenario=clean",
        f"data.feature_ablation_profile={feature_ablation_profile}",
        f"data.feature_ablation_seed={FEATURE_ABLATION_SEED}",
        f"data.candidate_seed={CANDIDATE_SEED}",
        f"training.seed={training_seed}",
        f"training.epochs={epochs}",
        "wandb.enabled=false",
        "experiment.wave=TASKA_EVAL",
        f"experiment.relation_ablation.enabled={str(relation_ablation_enabled).lower()}",
        f"experiment.relation_ablation.removed_relations={removed_override}",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name="config", overrides=overrides)


def collect_logits(
    model: torch.nn.Module,
    data: Any,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    data = data.to(device)

    with torch.no_grad():
        logits = model(data).view(-1)

    labels = data.edge_label.detach().cpu().float().view(-1).numpy()
    logits_np = logits.detach().cpu().float().numpy()
    probabilities = 1.0 / (1.0 + np.exp(-logits_np))
    return labels, logits_np, probabilities


def evaluate_probabilities(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    if len(np.unique(labels)) < 2:
        return {
            "auprc": float("nan"),
            "auroc": float("nan"),
            "brier": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "f2": float("nan"),
        }

    predictions = (probabilities >= threshold).astype(int)
    precision = float(precision_score(labels, predictions, zero_division=0))
    recall = float(recall_score(labels, predictions, zero_division=0))
    f1 = float(f1_score(labels, predictions, zero_division=0))
    beta = 2.0
    if precision + recall == 0:
        f2 = 0.0
    else:
        f2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)

    return {
        "auprc": float(average_precision_score(labels, probabilities)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": float(f2),
    }


def compare_logits(
    reference_logits: np.ndarray,
    perturbed_logits: np.ndarray,
) -> dict[str, float]:
    mean_absolute_change = float(
        np.mean(np.abs(reference_logits - perturbed_logits))
    )
    spearman = spearmanr(reference_logits, perturbed_logits).statistic
    return {
        "mean_abs_logit_change": mean_absolute_change,
        "spearman_logits": float(spearman),
    }


def select_threshold_on_validation(
    model: torch.nn.Module,
    valid_data: Any,
    device: torch.device,
) -> float:
    evaluator = ReconEvaluator(OmegaConf.create({"training": {}, "model": {"name": "tmp"}}))
    # Reuse SynEvaluator threshold search via a thin wrapper
    labels, _, probabilities = collect_logits(model, valid_data, device)
    if len(np.unique(labels)) < 2:
        return 0.5
    candidates = np.unique(np.quantile(probabilities, np.linspace(0.02, 0.98, 97)))
    best_f1, best_threshold = -1.0, 0.5
    for threshold in candidates:
        pred = (probabilities >= threshold).astype(int)
        score = f1_score(labels, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def _checkpoint_path(model_key: str, num_layers: int, seed: int) -> Path:
    return CHECKPOINT_ROOT / f"{model_key}_L{num_layers}_seed{seed}" / "empirical_best.pt"


def train_or_load_empirical_checkpoint(
    *,
    model_key: str,
    hydra_model: str,
    num_layers: int,
    seed: int,
    epochs: int = 100,
    force_retrain: bool = False,
) -> tuple[torch.nn.Module, Any, Any, Any, torch.device, DictConfig, float]:
    """Return model + empirical train/valid/test data + device + cfg + best valid AUPRC."""
    cfg = compose_task_a_cfg(
        hydra_model=hydra_model,
        num_layers=num_layers,
        training_seed=seed,
        feature_ablation_profile="empirical",
        epochs=epochs,
    )
    device = resolve_device(cfg)
    train_data, valid_data, test_data, _ = load_recon_heterodata(cfg)

    ckpt_path = _checkpoint_path(model_key, num_layers, seed)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg, train_data).to(device)
    model.grad_clip = float(getattr(cfg.training, "grad_clip", 0.0))

    if ckpt_path.exists() and not force_retrain:
        payload = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(payload["model_state_dict"])
        best_valid = float(payload.get("best_valid_auprc", float("nan")))
        return model, train_data, valid_data, test_data, device, cfg, best_valid

    set_seed(seed)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.training.lr),
        weight_decay=float(getattr(cfg.training, "weight_decay", 0.0)),
    )
    labels = train_data.edge_label.float()
    n_pos = float(labels.sum().clamp(min=1.0))
    n_neg = float((labels.numel() - labels.sum()).clamp(min=1.0))
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(n_neg / n_pos, device=device)
    )
    eval_criterion = torch.nn.BCEWithLogitsLoss()
    evaluator = ReconEvaluator(cfg)

    best_valid = -float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience = int(getattr(cfg.training, "early_stopping_patience", 20))
    wait = 0

    for _epoch in range(1, int(cfg.training.epochs) + 1):
        train_epoch(model, train_data, optimizer, criterion, device)
        valid_metrics = evaluator.evaluate(
            model, valid_data, eval_criterion, device, threshold=0.5
        )
        current = float(valid_metrics.get("auprc", float("nan")))
        if np.isfinite(current) and current > best_valid:
            best_valid = current
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if patience > 0 and wait >= patience:
                break

    model.load_state_dict(best_state)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "cfg": OmegaConf.to_container(cfg, resolve=True),
            "best_valid_auprc": best_valid,
            "model_key": model_key,
            "num_layers": num_layers,
            "training_seed": seed,
        },
        ckpt_path,
    )
    return model, train_data, valid_data, test_data, device, cfg, best_valid


def load_trained_empirical_model_and_test_data(
    *,
    model_name: str,
    num_layers: int,
    seed: int,
    epochs: int = 100,
    force_retrain: bool = False,
) -> tuple[torch.nn.Module, Any, Any, Any, torch.device]:
    """API expected by E1/E2 skeletons.

    Returns
    -------
    model, valid_data, test_data, train_data, device
    """
    mapping = {
        "hetero_sage": ("hetero_sage", "TaskA_hetero_sage", 2),
        "rgcn": ("rgcn", "TaskA_rgcn", 1),
        "TaskA_hetero_sage": ("hetero_sage", "TaskA_hetero_sage", 2),
        "TaskA_rgcn": ("rgcn", "TaskA_rgcn", 1),
    }
    if model_name not in mapping:
        raise ValueError(f"Unknown model_name={model_name!r}")

    model_key, hydra_model, default_layers = mapping[model_name]
    layers = int(num_layers) if num_layers is not None else default_layers

    model, train_data, valid_data, test_data, device, _cfg, _best = (
        train_or_load_empirical_checkpoint(
            model_key=model_key,
            hydra_model=hydra_model,
            num_layers=layers,
            seed=seed,
            epochs=epochs,
            force_retrain=force_retrain,
        )
    )
    return model, valid_data, test_data, train_data, device
