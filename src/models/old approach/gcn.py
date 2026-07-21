import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList
from torch_geometric.nn import GCNConv, global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()
        hidden_dim = cfg.model.hidden_dim
        num_layers = cfg.model.num_layers

        self.convs = ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.lin = Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        x = global_mean_pool(x, batch)
        x = self.lin(x).view(-1)
        return x