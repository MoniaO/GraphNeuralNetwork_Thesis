from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)

from evaluation.oversmoothing_metrics import compute_oversmoothing_metrics

OVERSMOOTHING_EXCLUDED_MODELS = {
    "linear",
    "linear_lp",
    "linear_hetero_lp",
}


class SynEvaluatorNode:

    def __init__(self, cfg, target_endpoint_names: Optional[List[str]] = None):
        self.cfg = cfg
        self.target_endpoint_names = target_endpoint_names or []
        self.per_endpoint = bool(getattr(cfg.training, "eval_per_endpoint", False))

        self.track_oversmoothing = bool(getattr(cfg.training, "track_oversmoothing", True))
        self.oversmoothing_sample_size = int(getattr(cfg.training, "oversmoothing_sample_size", 2000))

        model_name = str(getattr(cfg.model, "name", "")).lower()
        excluded = set(getattr(cfg.training, "oversmoothing_excluded_models", OVERSMOOTHING_EXCLUDED_MODELS))
        self.oversmoothing_applicable = model_name not in excluded

        self.classification_threshold = float(getattr(cfg.training, "eval_threshold", 0.5))
        self.auto_threshold = bool(getattr(cfg.training, "auto_threshold", True))
        self._threshold_quantiles = np.linspace(0.02, 0.98, 97)

    # ------------------------------------------------------------------
    # Iteracja po batchach (kluczowa roznica wzgledem starego evaluatora)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _forward_probs(self, model, loader: DataLoader, criterion, device):
        """Iteruje po WSZYSTKICH batchach loadera i agreguje y_true/y_prob,
        zamiast jednego forward() na cala populacje (jak przy jednym duzym
        grafie bipartite). Zwraca rowniez ostatni batch danych (do
        oversmoothing diagnostics) oraz sredni loss."""
        model.eval()
        all_y_true: List[np.ndarray] = []
        all_y_prob: List[np.ndarray] = []
        losses: List[float] = []
        last_batch = None

        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            labels = self._get_labels(model, batch)

            loss = criterion(logits, labels)
            losses.append(float(loss.item()))

            probs = torch.sigmoid(logits)
            all_y_true.append(labels.detach().cpu().numpy())
            all_y_prob.append(probs.detach().cpu().numpy())
            last_batch = batch

        y_true = np.concatenate(all_y_true) if all_y_true else np.array([])
        y_prob = np.concatenate(all_y_prob) if all_y_prob else np.array([])
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        return last_batch, y_true, y_prob, mean_loss

    def _get_labels(self, model, batch) -> torch.Tensor:
        """Etykiety w tej samej kolejnosci co logity z model.forward()."""
        from models.TaskB.gnn_node_clf_simple_targeted import get_targeted_labels
        return get_targeted_labels(
            batch, model.target_local_idx, target_node_type=model.target_node_type
        )

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
    def select_threshold(self, model, valid_loader: DataLoader, device) -> float:
        if not self.auto_threshold:
            return self.classification_threshold
        _, y_true, y_prob, _ = self._forward_probs(
            model, valid_loader, torch.nn.BCEWithLogitsLoss(), device
        )
        return self._best_f1_threshold(y_true, y_prob)

    # ------------------------------------------------------------------
    # Ewaluacja glowna
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, model, loader: DataLoader, criterion, device, threshold: Optional[float] = None) -> dict:
        last_batch, y_true, y_prob, loss = self._forward_probs(model, loader, criterion, device)
        active_threshold = self.classification_threshold if threshold is None else float(threshold)

        metrics = self._classification_metrics(y_true, y_prob, prefix="", threshold=active_threshold)
        metrics["loss"] = float(loss)
        metrics["classification_threshold"] = active_threshold

        # Rozbicie per-endpoint: kazdy endpoint to kolumna w ukladzie
        # [n_graphs_in_batch * n_targets], wiec rozdzielamy modulo n_targets
        # - dużo prostsze niz stary edge_label_target_idx.
        if self.per_endpoint and self.target_endpoint_names:
            n_targets = len(self.target_endpoint_names)
            if len(y_true) % n_targets == 0 and len(y_true) > 0:
                y_true_reshaped = y_true.reshape(-1, n_targets)
                y_prob_reshaped = y_prob.reshape(-1, n_targets)
                for i, name in enumerate(self.target_endpoint_names):
                    yt = y_true_reshaped[:, i]
                    yp = y_prob_reshaped[:, i]
                    metrics.update(
                        self._classification_metrics(yt, yp, prefix=f"{name}_", threshold=active_threshold)
                    )

        if (
            self.track_oversmoothing
            and self.oversmoothing_applicable
            and hasattr(model, "encode")
            and last_batch is not None
        ):
            z_dict = model.encode(last_batch)
            metrics.update(
                compute_oversmoothing_metrics(z_dict, last_batch, sample_size=self.oversmoothing_sample_size)
            )

        return metrics

    def _classification_metrics(
        self, y_true: np.ndarray, y_prob: np.ndarray, prefix: str, threshold: float
    ) -> dict:
        n = len(y_true)
        positives = int(y_true.sum()) if n > 0 else 0
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
