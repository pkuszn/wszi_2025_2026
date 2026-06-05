import torch.nn as nn
import torch
from torch_geometric.nn import GINConv, global_add_pool, BatchNorm

class HybridModel(nn.Module):
    def __init__(self, num_node_features, num_extra_features, hidden_channels=64):
        """Initialization.

        Args:
            num_node_features (_type_): Number of features
            num_extra_features (_type_): Number of global features
            hidden_channels (int, optional): The number of channels in the internal layers. Defaults to 64.
        """
        super().__init__()

        nn1 = nn.Sequential(
            nn.Linear(num_node_features, hidden_channels),
            nn.ReLU(), 
            nn.Linear(hidden_channels, hidden_channels)
        ) # Initializes the module in sequential order

        self.conv1 = GINConv(nn1)
        self.bn1 = BatchNorm(hidden_channels)

        nn2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), 
            nn.ReLU(), 
            nn.Linear(hidden_channels, hidden_channels)
        )

        self.conv2 = GINConv(nn2)
        self.bn2 = BatchNorm(hidden_channels)

        self.mlp = nn.Sequential(
            nn.Linear(num_extra_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_channels + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x, edge_index, batch, extra_features):
        """Roadmap for data to reach a prediction

        Args:
            x (_type_): Nodes
            edge_index (_type_): Connectivity, graph tolopogy
            batch (_type_): molecule mapping
            extra_features (_type_): global descriptors, physicochemical properties
        """
        x = self.conv1(x, edge_index).relu()
        x = self.bn1(x)
        x = self.conv2(x, edge_index).relu()
        x = self.bn2(x)
        x = global_add_pool(x, batch)

        extra = self.mlp(extra_features)
        
        combined = torch.cat([x, extra], dim=1)

        return self.regressor(combined)
