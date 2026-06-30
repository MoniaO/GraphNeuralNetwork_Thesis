import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, cfg, num_nodes):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_nodes, cfg.model.hidden_dim)
        self.conv1 = GCNConv(cfg.model.hidden_dim, cfg.model.hidden_dim)
        self.conv2 = GCNConv(cfg.model.hidden_dim, cfg.model.hidden_dim)

    def forward(self, edge_index):
        x = self.embedding.weight
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def predict(self, z, edge):
        return (z[edge[:, 0]] * z[edge[:, 1]]).sum(dim=-1)