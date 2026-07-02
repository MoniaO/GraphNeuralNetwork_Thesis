import torch
import numpy as np
from ogb.linkproppred import Evaluator
from sklearn.metrics import roc_auc_score, average_precision_score


class OGBEvaluator:
    def __init__(self, cfg):
        self.evaluator = Evaluator(name=cfg.data.name)

    @torch.no_grad()
    def evaluate(self, model, graph, split_idx, device):
        model.eval()
        x = graph.x.to(device) if graph.x is not None else None
        edge_index = graph.edge_index.to(device)
        z = model(edge_index)

        results = {}
        for split in ['train', 'valid', 'test']:
            pos_edge = split_idx[split]['edge'].to(device)
            
            if "edge_neg" in split_idx[split]:
                neg_edge = split_idx[split]["edge_neg"].to(device)
            else:
                neg_edge = torch.randint(0, graph.num_nodes, pos_edge.shape, device=device)

            pos_score = model.predict(z, pos_edge).cpu()
            neg_score = model.predict(z, neg_edge).cpu()

            # etykiety i score
            scores = torch.cat([pos_score, neg_score]).numpy()
            labels = torch.cat([
                torch.ones(pos_score.size(0)),
                torch.zeros(neg_score.size(0))
            ]).numpy()

            # ROC-AUC
            auc = roc_auc_score(labels, scores)

            # AUPRC
            auprc = average_precision_score(labels, scores)

            if split != "train":
                hits = self.evaluator.eval({
                    "y_pred_pos": pos_score,
                    "y_pred_neg": neg_score,
                })["hits@20"]
                results[f"{split}/hits@20"] = float(hits)

            results[f'{split}/auc']     = float(auc)
            results[f'{split}/auprc']   = float(auprc)

        return results