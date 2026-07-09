import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList
from torch_geometric.nn import GATConv, global_mean_pool


class GATNet(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()
        hidden_dim = cfg.model.hidden_dim
        num_layers = cfg.model.num_layers
        heads = cfg.model.heads

        self.convs = ModuleList()

        self.convs.append(GATConv(in_channels, hidden_dim, heads=heads, concat=False))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=heads, concat=False))

        self.lin = Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        x = global_mean_pool(x, batch)
        x = self.lin(x).view(-1)
        return x