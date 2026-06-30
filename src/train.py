import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from src.data.load_data import load_dataset, get_loaders

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def train(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=f"{cfg.model.name}_{cfg.data.name}",
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    dataset, split_idx = load_dataset(cfg)
    train_loader, val_loader = get_loaders(
        dataset, split_idx, cfg.training.batch_size
    )

    # TODO: model
    # TODO: training

    wandb.finish()

if __name__ == "__main__":
    train()