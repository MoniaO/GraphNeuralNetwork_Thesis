# import hydra
# import torch
# import wandb
# from omegaconf import DictConfig, OmegaConf

# from data.load_data import load_dataset, get_loaders
# from models import build_model
# from evaluation import build_evaluator


# def train_epoch(model, graph, train_edges, optimizer, device):
#     model.train()
#     optimizer.zero_grad()

#     x = graph.x.to(device) if graph.x is not None else None
#     edge_index = graph.edge_index.to(device)
#     z = model(edge_index)

#     pos_edge = train_edges.to(device)
#     neg_edge = torch.randint(0, graph.num_nodes, pos_edge.shape, device=device)

#     pos_score = model.predict(z, pos_edge)
#     neg_score = model.predict(z, neg_edge)

#     loss = -torch.log(pos_score.sigmoid() + 1e-15).mean() \
#            -torch.log(1 - neg_score.sigmoid() + 1e-15).mean()

#     loss.backward()
#     optimizer.step()

#     return loss.item()


# @hydra.main(version_base=None, config_path="../configs", config_name="config")
# def train(cfg: DictConfig):
#     print(OmegaConf.to_yaml(cfg))

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     wandb.init(
#         project=cfg.wandb.project,
#         entity=cfg.wandb.entity,
#         name=cfg.wandb.run_name if "run_name" in cfg.wandb else f"{cfg.model.name}_{cfg.data.name}_epochs{cfg.training.epochs}_lr{cfg.training.lr}_{cfg.training.optimizer}",
#         config=OmegaConf.to_container(cfg, resolve=True),
#     )

#     dataset, split_idx = load_dataset(cfg)
#     graph, train_loader, val_loader = get_loaders(dataset, split_idx, cfg)

#     #in_channels = graph.x.shape[1] if graph.x is not None else None
#     model = build_model(cfg, num_nodes=graph.num_nodes).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

#     evaluator = build_evaluator(cfg)

#     best_valid = -1
#     best_epoch = -1
#     best_state = None

#     for epoch in range(1, cfg.training.epochs + 1):
#         loss = train_epoch(model, graph, split_idx["train"]["edge"], optimizer, device)
#         metrics = evaluator.evaluate(model, graph, split_idx, device)

#         log_dict = {
#             "epoch": epoch,
#             "loss": loss,
#             "lr": optimizer.param_groups[0]["lr"],
#             **metrics,
#         }
#         wandb.log(log_dict, step=epoch)

#         valid_score = metrics.get("valid/auprc", metrics.get("valid/hits@20", -1))
#         if valid_score > best_valid:
#             best_valid = valid_score
#             best_epoch = epoch
#             best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

#         print(
#             f"Epoch {epoch:03d} | "
#             f"loss {loss:.4f} | "
#             f"train AUC {metrics.get('train/auc', float('nan')):.4f} | "
#             f"train AUPRC {metrics.get('train/auprc', float('nan')):.4f} | "
#             f"valid AUC {metrics.get('valid/auc', float('nan')):.4f} | "
#             f"valid AUPRC {metrics.get('valid/auprc', float('nan')):.4f}"
#             f"valid hits@20 {metrics.get('valid/hits@20', float('nan')):.4f}"
#             f"test AUC {metrics.get('test/auc', float('nan')):.4f} | "
#             f"test AUPRC {metrics.get('test/auprc', float('nan')):.4f}"
#             f"test hits@20 {metrics.get('test/hits@20', float('nan')):.4f}"
#         )

#     if best_state is not None:
#         model.load_state_dict(best_state)

#     wandb.summary["best_epoch"] = best_epoch
#     wandb.summary["best_valid"] = best_valid
#     wandb.finish()


# if __name__ == "__main__":
#     train()


import hydra
import torch
import wandb
import torch.nn.functional as F
import torch_geometric.transforms as T

from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import negative_sampling, to_undirected

from data.load_data import load_dataset
from models import build_model


def train_epoch(model, train_data, optimizer, device):
    model.train()
    optimizer.zero_grad()

    edge_index = train_data.edge_index.to(device)
    z = model(edge_index)

    pos_edge = train_data.edge_label_index.t().to(device)

    neg_edge = negative_sampling(
        edge_index=edge_index,
        num_nodes=train_data.num_nodes,
        num_neg_samples=pos_edge.size(0),
        method="sparse",
    ).t()

    pos_score = model.predict(z, pos_edge)
    neg_score = model.predict(z, neg_edge)

    scores = torch.cat([pos_score, neg_score], dim=0)
    labels = torch.cat(
        [
            torch.ones(pos_score.size(0), device=device),
            torch.zeros(neg_score.size(0), device=device),
        ],
        dim=0,
    )

    loss = F.binary_cross_entropy_with_logits(scores, labels)
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model, data, device):
    model.eval()

    edge_index = data.edge_index.to(device)
    z = model(edge_index)

    edge = data.edge_label_index.t().to(device)
    y_true = data.edge_label.float().to(device)

    logits = model.predict(z, edge)
    y_score = torch.sigmoid(logits)

    y_true_np = y_true.cpu().numpy()
    y_score_np = y_score.cpu().numpy()

    auc = roc_auc_score(y_true_np, y_score_np)
    ap = average_precision_score(y_true_np, y_score_np)

    return {
        "auc": auc,
        "ap": ap,
    }


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.run_name
        if "run_name" in cfg.wandb and cfg.wandb.run_name is not None
        else f"{cfg.model.name}_{cfg.data.name}",
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    data = load_dataset(cfg)

    if cfg.training.to_undirected:
        data.edge_index = to_undirected(data.edge_index)

    transform = T.RandomLinkSplit(
        num_val=cfg.training.num_val,
        num_test=cfg.training.num_test,
        is_undirected=cfg.training.to_undirected,
        add_negative_train_samples=False,
        neg_sampling_ratio=1.0,
    )

    train_data, val_data, test_data = transform(data)

    model = build_model(cfg, num_nodes=train_data.num_nodes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

    best_val_auc = -1.0
    best_epoch = -1
    best_state = None

    for epoch in range(1, cfg.training.epochs + 1):
        loss = train_epoch(model, train_data, optimizer, device)
        val_metrics = evaluate(model, val_data, device)
        test_metrics = evaluate(model, test_data, device)

        log_dict = {
            "epoch": epoch,
            "train/loss": loss,
            "val/auc": val_metrics["auc"],
            "val/ap": val_metrics["ap"],
            "test/auc": test_metrics["auc"],
            "test/ap": test_metrics["ap"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        wandb.log(log_dict, step=epoch)

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        print(
            f"Epoch {epoch:03d} | "
            f"loss {loss:.4f} | "
            f"val AUC {val_metrics['auc']:.4f} | "
            f"val AP {val_metrics['ap']:.4f} | "
            f"test AUC {test_metrics['auc']:.4f} | "
            f"test AP {test_metrics['ap']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    wandb.summary["best_epoch"] = best_epoch
    wandb.summary["best_val_auc"] = best_val_auc

    wandb.finish()


if __name__ == "__main__":
    train()