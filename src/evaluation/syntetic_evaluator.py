import torch
import numpy as np
from ogb.linkproppred import Evaluator
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    brier_score_loss
)


class SynEvaluator:
    def __init__(self, cfg):
        self.evaluator = Evaluator(name=cfg.data.name)

    @torch.no_grad()
    def evaluate(self, model, graph, split_idx, device):
        model.eval()
        edge_index = graph.edge_index.to(device)
        batch = graph.batch.to(device) if hasattr(graph, 'batch') else None
        z = model(graph.x.to(device), edge_index, batch)

        results = {}

        for split in ["train", "valid", "test"]:
            pos_edge = split_idx[split]["edge"].to(device)

            if "edge_neg" in split_idx[split]:
                neg_edge = split_idx[split]["edge_neg"].to(device)
            else:
                neg_edge = torch.randint(
                    0, graph.num_nodes, pos_edge.shape, device=device
                )

            pos_score = model.predict(z, pos_edge).view(-1)
            neg_score = model.predict(z, neg_edge).view(-1)

            scores = torch.cat([pos_score, neg_score], dim=0)
            probs = torch.sigmoid(scores).cpu().numpy()

            labels = torch.cat([
                torch.ones(pos_score.size(0)),
                torch.zeros(neg_score.size(0))
            ]).cpu().numpy()

            if len(np.unique(labels)) < 2:
                auc_roc = np.nan
                auprc = np.nan
                brier = np.nan
            else:
                auc_roc = roc_auc_score(labels, probs)

                precision, recall, _ = precision_recall_curve(labels, probs)
                auprc = auc(recall, precision)

                brier = brier_score_loss(labels, probs)

            results[f"{split}/auc"] = float(auc_roc) if not np.isnan(auc_roc) else np.nan
            results[f"{split}/auprc"] = float(auprc) if not np.isnan(auprc) else np.nan
            results[f"{split}/brier"] = float(brier) if not np.isnan(brier) else np.nan

        return results