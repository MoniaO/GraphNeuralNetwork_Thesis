from models.gcn import GCN
from models.sage import GraphSAGE
from models.gat import GAT

def build_model(cfg, num_nodes):
    models = {
        "gcn":  GCN,
        "sage": GraphSAGE,
        "gat":  GAT,
    }
    if cfg.model.name not in models:
        raise ValueError(f"Uknown model: {cfg.model.name}")
    return models[cfg.model.name](cfg, num_nodes)