import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss


class SynEvaluator:
    def __init__(self, cfg):
        self.cfg = cfg

    @torch.no_grad()
    def evaluate(self, model, loader, criterion, device):
        model.eval()

        total_loss = 0.0
        total_graphs = 0
        y_true_all = []
        y_prob_all = []

        for batch in loader:
            batch = batch.to(device)

            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(logits, batch.y.float())

            probs = torch.sigmoid(logits)

            total_loss += loss.item() * batch.num_graphs
            total_graphs += batch.num_graphs

            y_true_all.append(batch.y.detach().cpu())
            y_prob_all.append(probs.detach().cpu())

        y_true = torch.cat(y_true_all).numpy()
        y_prob = torch.cat(y_prob_all).numpy()

        metrics = {
            "loss": total_loss / total_graphs,
            "auc": np.nan,
            "auprc": np.nan,
            "brier": np.nan,
        }

        if len(np.unique(y_true)) > 1:
            metrics["auc"] = roc_auc_score(y_true, y_prob)

            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            metrics["auprc"] = auc(recall, precision)

            metrics["brier"] = brier_score_loss(y_true, y_prob)

        return metrics