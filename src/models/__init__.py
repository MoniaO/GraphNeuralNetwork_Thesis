from models.gcn import GCN
from models.sage import GraphSAGE
from models.gat import GATNet
from models.gin import GINNet
from models.gnn_lp import SimpleHeteroGNN


def build_model(cfg, in_channels):
    models = {
        "gcn":  GCN ,
        "sage": GraphSAGE,
        "gat":  GATNet,
        "gin":  GINNet, 
        "gnn_lp":  SimpleHeteroGNN
    }

    if cfg.model.name == "gcn":
        return GCN(cfg, in_channels)
    elif cfg.model.name == "sage":
        return GraphSAGE(cfg, in_channels)
    elif cfg.model.name == "gin":
        return GINNet(cfg, in_channels)
    elif cfg.model.name == "gat":
        return GATNet(cfg, in_channels)
    elif cfg.model.name == "gnn_lp":
        return SimpleHeteroGNN(cfg, in_channels)
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")