import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, ModuleList
from torch_geometric.nn import GINConv, global_mean_pool


class GIN(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()
        hidden_dim = cfg.model.hidden_dim
        num_layers = cfg.model.num_layers

        self.convs = ModuleList()

        mlp_in = Sequential(
            Linear(in_channels, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
        )
        self.convs.append(GINConv(mlp_in))

        for _ in range(num_layers - 1):
            mlp_hidden = Sequential(
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp_hidden))

        self.lin = Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        x = global_mean_pool(x, batch)
        x = self.lin(x).view(-1)
        return x