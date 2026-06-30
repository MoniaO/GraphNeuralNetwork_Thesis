import torch
import torch.nn.functional as F
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from ogb.linkproppred import Evaluator
from data.load_data import load_dataset, get_loaders
from models import build_model

def train_epoch(model, graph, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        z = model(graph.edge_index.to(device))

        pos_edge = batch.to(device)
        neg_edge = torch.randint(0, graph.num_nodes, pos_edge.shape, device=device)

        pos_score = model.predict(z, pos_edge)
        neg_score = model.predict(z, neg_edge)

        loss = -torch.log(pos_score.sigmoid() + 1e-15).mean() \
               -torch.log(1 - neg_score.sigmoid() + 1e-15).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


@torch.no_grad()
def evaluate(model, graph, evaluator, split_idx, device):
    model.eval()
    z = model(graph.edge_index.to(device))
    results = {}
    for split in ['train', 'valid', 'test']:
        pos_edge = split_idx[split]['edge'].to(device)
        neg_edge = split_idx[split]['edge_neg'].to(device)
        pos_score = model.predict(z, pos_edge)
        neg_score = model.predict(z, neg_edge)
        results[split] = evaluator.eval({
            'y_pred_pos': pos_score,
            'y_pred_neg': neg_score,
        })['hits@20']
    return results


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def train(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=f"{cfg.model.name}_{cfg.data.name}",
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    dataset, split_idx = load_dataset(cfg)
    graph, train_loader, val_loader = get_loaders(dataset, split_idx, cfg)
    graph = graph.to(device)

    evaluator = Evaluator(name=cfg.data.name)
    model = build_model(cfg, num_nodes=graph.num_nodes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

    for epoch in range(1, cfg.training.epochs + 1):
        loss = train_epoch(model, graph, train_loader, optimizer, device)
        results = evaluate(model, graph, evaluator, split_idx, device)

        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} "
              f"| Train hits@20: {results['train']:.4f} "
              f"| Val hits@20: {results['valid']:.4f}")

        wandb.log({
            "epoch": epoch,
            "loss": loss,
            "train/hits@20": results['train'],
            "val/hits@20":   results['valid'],
            "test/hits@20":  results['test'],
        })

    wandb.finish()


if __name__ == "__main__":
    train()