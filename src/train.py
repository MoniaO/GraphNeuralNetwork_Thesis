import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from data.load_data import load_dataset, get_loaders
from models import build_model
from evaluation import build_evaluator


def train_epoch(model, graph, train_edges, optimizer, device):
    model.train()
    optimizer.zero_grad()

    x = graph.x.to(device) if graph.x is not None else None
    edge_index = graph.edge_index.to(device)
    z = model(edge_index)

    pos_edge = train_edges.to(device)
    neg_edge = torch.randint(0, graph.num_nodes, pos_edge.shape, device=device)

    pos_score = model.predict(z, pos_edge)
    neg_score = model.predict(z, neg_edge)

    loss = -torch.log(pos_score.sigmoid() + 1e-15).mean() \
           -torch.log(1 - neg_score.sigmoid() + 1e-15).mean()

    loss.backward()
    optimizer.step()

    return loss.item()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.run_name if "run_name" in cfg.wandb else f"{cfg.model.name}_{cfg.data.name}_epochs{cfg.training.epochs}_lr{cfg.training.lr}_{cfg.training.optimizer}",
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    dataset, split_idx = load_dataset(cfg)
    graph, train_loader, val_loader = get_loaders(dataset, split_idx, cfg)

    #in_channels = graph.x.shape[1] if graph.x is not None else None
    model = build_model(cfg, num_nodes=graph.num_nodes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

    evaluator = build_evaluator(cfg)

    best_valid = -1
    best_epoch = -1
    best_state = None

    for epoch in range(1, cfg.training.epochs + 1):
        loss = train_epoch(model, graph, split_idx["train"]["edge"], optimizer, device)
        metrics = evaluator.evaluate(model, graph, split_idx, device)

        log_dict = {
            "epoch": epoch,
            "loss": loss,
            "lr": optimizer.param_groups[0]["lr"],
            **metrics,
        }
        wandb.log(log_dict, step=epoch)

        valid_score = metrics.get("valid/auprc", metrics.get("valid/hits@20", -1))
        if valid_score > best_valid:
            best_valid = valid_score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:03d} | "
            f"loss {loss:.4f} | "
            f"train AUC {metrics.get('train/auc', float('nan')):.4f} | "
            f"train AUPRC {metrics.get('train/auprc', float('nan')):.4f} | "
            f"valid AUC {metrics.get('valid/auc', float('nan')):.4f} | "
            f"valid AUPRC {metrics.get('valid/auprc', float('nan')):.4f}"
            f"valid hits@20 {metrics.get('valid/hits@20', float('nan')):.4f}"
            f"test AUC {metrics.get('test/auc', float('nan')):.4f} | "
            f"test AUPRC {metrics.get('test/auprc', float('nan')):.4f}"
            f"test hits@20 {metrics.get('test/hits@20', float('nan')):.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    wandb.summary["best_epoch"] = best_epoch
    wandb.summary["best_valid"] = best_valid
    wandb.finish()


if __name__ == "__main__":
    train()