
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

from evaluation.oversmoothing_metrics import compute_oversmoothing_metrics


class SynEvaluator:
    """Evaluator for link prediction: ranking/calibration/classification
    metrics. Oversmoothing diagnostics now live in oversmoothing_metrics.py
    and are imported here, not implemented inline.
    """

    def __init__(self, cfg, node_to_idx=None):
        self.cfg = cfg
        self.node_to_idx = node_to_idx or {}
        self.idx_to_node = {v: k for k, v in self.node_to_idx.items()}
        self.per_endpoint = bool(getattr(cfg.training, "eval_per_endpoint", False))
        self.track_oversmoothing = bool(getattr(cfg.training, "track_oversmoothing", True))
        self.oversmoothing_sample_size = int(getattr(cfg.training, "oversmoothing_sample_size", 2000))
        self.classification_threshold = float(getattr(cfg.training, "eval_threshold", 0.5))

    @torch.no_grad()
    def evaluate(self, model, data, criterion, device):
        model.eval()
        data = data.to(device)
        logits = model(data)
        labels = data[("patient", "has_adr", "variable")].edge_label.float()
        probs = torch.sigmoid(logits)

        loss = criterion(logits, labels).item()
        y_true = labels.detach().cpu().numpy()
        y_prob = probs.detach().cpu().numpy()

        metrics = self._classification_metrics(y_true, y_prob, prefix="")
        metrics["loss"] = float(loss)

        if self.per_endpoint:
            target_idx = data[("patient", "has_adr", "variable")].edge_label_target_idx.cpu().numpy()
            unique_targets = np.unique(target_idx)
            if len(unique_targets) > 1:
                for t in unique_targets:
                    mask = target_idx == t
                    yt, yp = y_true[mask], y_prob[mask]
                    name = self.idx_to_node.get(int(t), str(t))
                    metrics.update(self._classification_metrics(yt, yp, prefix=f"{name}_"))

        if self.track_oversmoothing and hasattr(model, "encode"):
            z_dict = model.encode(data)
            metrics.update(
                compute_oversmoothing_metrics(z_dict, data, sample_size=self.oversmoothing_sample_size)
            )

        return metrics

    def _classification_metrics(self, y_true: np.ndarray, y_prob: np.ndarray, prefix: str) -> dict:
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

        y_pred = (y_prob >= self.classification_threshold).astype(int)
        out[f"{prefix}f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        out[f"{prefix}precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        out[f"{prefix}recall"] = float(recall_score(y_true, y_pred, zero_division=0))

        return out
