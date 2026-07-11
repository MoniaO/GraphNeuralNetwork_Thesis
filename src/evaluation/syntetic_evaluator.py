from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


class SynEvaluator:
    def __init__(self, cfg):
        self.cfg = cfg

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

        metrics = {
            "loss": float(loss),
            "auc": float("nan"),
            "auprc": float("nan"),
            "brier": float("nan"),
        }

        if len(np.unique(y_true)) > 1:
            metrics["auc"] = float(roc_auc_score(y_true, y_prob))
            metrics["auprc"] = float(average_precision_score(y_true, y_prob))
            metrics["brier"] = float(brier_score_loss(y_true, y_prob))

        return metrics
