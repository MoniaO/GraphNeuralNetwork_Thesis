# from __future__ import annotations

# import numpy as np
# import torch
# from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


# class SynEvaluator:
#     def __init__(self, cfg):
#         self.cfg = cfg

#     @torch.no_grad()
#     def evaluate(self, model, data, criterion, device):
#         model.eval()
#         data = data.to(device)

#         logits = model(data)
#         labels = data[("patient", "has_adr", "variable")].edge_label.float()
#         probs = torch.sigmoid(logits)

#         loss = criterion(logits, labels).item()
#         y_true = labels.detach().cpu().numpy()
#         y_prob = probs.detach().cpu().numpy()

#         metrics = {
#             "loss": float(loss),
#             "auc": float("nan"),
#             "auprc": float("nan"),
#             "brier": float("nan"),
#         }

#         if len(np.unique(y_true)) > 1:
#             metrics["auc"] = float(roc_auc_score(y_true, y_prob))
#             metrics["auprc"] = float(average_precision_score(y_true, y_prob))
#             metrics["brier"] = float(brier_score_loss(y_true, y_prob))

#         return metrics

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

class SynEvaluator:
    def __init__(self, cfg, node_to_idx=None):
        self.cfg = cfg
        self.node_to_idx = node_to_idx or {}
        self.idx_to_node = {v: k for k, v in self.node_to_idx.items()}
        self.per_endpoint = bool(getattr(cfg.training, "eval_per_endpoint", False))

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

        metrics = {"loss": float(loss), "auc": float("nan"), "auprc": float("nan"), "brier": float("nan")}
        if len(np.unique(y_true)) > 1:
            metrics["auc"] = float(roc_auc_score(y_true, y_prob))
            metrics["auprc"] = float(average_precision_score(y_true, y_prob))
            metrics["brier"] = float(brier_score_loss(y_true, y_prob))

        if self.per_endpoint:
            target_idx = data[("patient", "has_adr", "variable")].edge_label_target_idx.cpu().numpy()
            for t in np.unique(target_idx):
                mask = target_idx == t
                yt, yp = y_true[mask], y_prob[mask]
                name = self.idx_to_node.get(int(t), str(t))
                if len(np.unique(yt)) > 1:
                    metrics[f"auc_{name}"] = float(roc_auc_score(yt, yp))
                    metrics[f"auprc_{name}"] = float(average_precision_score(yt, yp))
                else:
                    metrics[f"auc_{name}"] = float("nan")
                    metrics[f"auprc_{name}"] = float("nan")

        return metrics