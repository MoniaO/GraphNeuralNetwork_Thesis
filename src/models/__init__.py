from models.gcn import GCN
from models.sage import GraphSAGE
from models.gat import GATNet
from models.gin import GINNet


def build_model(cfg, in_channels):
    models = {
        "gcn":  GCN ,
        "sage": GraphSAGE,
        "gat":  GATNet,
        "gin":  GINNet
    }

    if cfg.model.name == "gcn":
        return GCN(cfg, in_channels)
    elif cfg.model.name == "sage":
        return GraphSAGE(cfg, in_channels)
    elif cfg.model.name == "gin":
        return GINNet(cfg, in_channels)
    elif cfg.model.name == "gat":
        return GATNet(cfg, in_channels)
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")