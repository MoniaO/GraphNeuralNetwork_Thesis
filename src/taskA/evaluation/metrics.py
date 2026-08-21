
"""Link-prediction metrics. FINAL model selection: valid AUPRC only.

AUROC, Brier, F1 are logged but do not rank the checkpoint.
Classification threshold comes from validation (`auto_threshold`).
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)

from taskA.evaluation.oversmoothing import compute_oversmoothing_metrics

# Model names for which oversmoothing diagnostics are not meaningful
# (e.g. linear models with no message passing / no notion of "layers").
OVERSMOOTHING_EXCLUDED_MODELS = {
    "linear",
    "linear_lp",
    "linear_hetero_lp",
}


class SynEvaluator:
    """Shared metrics for Task A structural link prediction.

    Subclasses provide `_forward_probs` for the concrete graph representation.
    Oversmoothing diagnostics live in `oversmoothing_metrics.py`.

    Classification threshold is no longer fixed at 0.5 by default: call
    `select_threshold(...)` on the validation split once per epoch, then
    pass the returned value into `evaluate(..., threshold=...)` for all
    splits (train/valid/test) so metrics are comparable within an epoch.
    """

    def __init__(self, cfg, node_to_idx=None):
        self.cfg = cfg
        self.node_to_idx = node_to_idx or {}
        self.idx_to_node = {v: k for k, v in self.node_to_idx.items()}

        self.track_oversmoothing = bool(getattr(cfg.training, "track_oversmoothing", True))
        self.oversmoothing_sample_size = int(getattr(cfg.training, "oversmoothing_sample_size", 2000))

        model_name = str(getattr(cfg.model, "name", "")).lower()
        excluded = set(getattr(cfg.training, "oversmoothing_excluded_models", OVERSMOOTHING_EXCLUDED_MODELS))
        self.oversmoothing_applicable = model_name not in excluded

        # Fallback threshold, used only if no threshold is explicitly selected/passed.
        self.classification_threshold = float(getattr(cfg.training, "eval_threshold", 0.5))

        # Whether to auto-select the best F1 threshold on validation (recommended: True).
        self.auto_threshold = bool(getattr(cfg.training, "auto_threshold", True))
        self._threshold_quantiles = np.linspace(0.02, 0.98, 97)

    @torch.no_grad()
    def _forward_probs(self, model, data, criterion, device):
        raise NotImplementedError("Task A evaluators must implement _forward_probs")

    def _best_f1_threshold(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        if len(np.unique(y_true)) < 2:
            return self.classification_threshold
        candidates = np.unique(np.quantile(y_prob, self._threshold_quantiles))
        best_f1, best_threshold = -1.0, self.classification_threshold
        for threshold in candidates:
            pred = (y_prob >= threshold).astype(int)
            f1 = f1_score(y_true, pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(threshold)
        return best_threshold

    @torch.no_grad()
    def select_threshold(self, model, valid_data, device) -> float:
        """Find the F1-optimal classification threshold on a validation split.

        Call this once per epoch on valid_data, then pass the result to
        `evaluate(..., threshold=...)` for train/valid/test so all splits
        use the same, validation-selected threshold within that epoch.
        """
        if not self.auto_threshold:
            return self.classification_threshold
        _, y_true, y_prob, _ = self._forward_probs(model, valid_data, torch.nn.BCEWithLogitsLoss(), device)
        return self._best_f1_threshold(y_true, y_prob)

    @torch.no_grad()
    def evaluate(self, model, data, criterion, device, threshold: float | None = None) -> dict:
        data, y_true, y_prob, loss = self._forward_probs(model, data, criterion, device)
        active_threshold = self.classification_threshold if threshold is None else float(threshold)

        metrics = self._classification_metrics(y_true, y_prob, prefix="", threshold=active_threshold)
        metrics["loss"] = float(loss)
        metrics["classification_threshold"] = active_threshold

        if self.track_oversmoothing and self.oversmoothing_applicable and hasattr(model, "encode"):
            z_dict = model.encode(data)
            metrics.update(
                compute_oversmoothing_metrics(z_dict, data, sample_size=self.oversmoothing_sample_size)
            )

        return metrics

    def _classification_metrics(
        self, y_true: np.ndarray, y_prob: np.ndarray, prefix: str, threshold: float
    ) -> dict:
        n = len(y_true)
        positives = int(y_true.sum())
        negatives = n - positives

        out = {
            f"{prefix}auc": float("nan"),
            f"{prefix}auprc": float("nan"),
            f"{prefix}brier": float("nan"),
            f"{prefix}f1": float("nan"),
            f"{prefix}precision": float("nan"),
            f"{prefix}recall": float("nan"),
            f"{prefix}positives": positives,
            f"{prefix}negatives": negatives,
            f"{prefix}auprc_baseline": float(positives / n) if n > 0 else float("nan"),
            f"{prefix}auprc_lift": float("nan"),
        }

        if positives == 0 or negatives == 0:
            return out

        out[f"{prefix}auc"] = float(roc_auc_score(y_true, y_prob))
        out[f"{prefix}auprc"] = float(average_precision_score(y_true, y_prob))
        out[f"{prefix}brier"] = float(brier_score_loss(y_true, y_prob))

        baseline = out[f"{prefix}auprc_baseline"]
        if baseline > 0:
            out[f"{prefix}auprc_lift"] = float(out[f"{prefix}auprc"] / baseline)

        y_pred = (y_prob >= threshold).astype(int)
        out[f"{prefix}f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        out[f"{prefix}precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        out[f"{prefix}recall"] = float(recall_score(y_true, y_pred, zero_division=0))

        return out
