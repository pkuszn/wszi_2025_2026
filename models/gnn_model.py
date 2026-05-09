import mlflow
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, global_add_pool

class GNNModel(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, model_type="GCN"):
        super().__init__()
        self.model_type = model_type
        
        if model_type == "GCN":
            self.conv1 = GCNConv(num_node_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
        elif model_type == "GIN":
            nn1 = torch.nn.Sequential(
                torch.nn.Linear(num_node_features, hidden_channels), 
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels)
            )
            self.conv1 = GINConv(nn1)
            
            nn2 = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels), 
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels)
            )
            self.conv2 = GINConv(nn2)

        self.lin = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()

        x = global_add_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)
        return self.lin(x)

    def train_gnn(self, model, loader, optimizer, device, mf_manager, run_name):
        model.to(device)
        criterion = torch.nn.MSELoss()
        
        with mf_manager.start_run(run_name=run_name):
            mlflow.log_param("model_type", model.model_type)
            mlflow.log_param("hidden_channels", model.conv2.out_channels if hasattr(model.conv2, 'out_channels') else "GIN_internal")
            
            model.train()
            total_loss = 0
            for data in loader:
                data = data.to(device)
                optimizer.zero_grad()
                out = model(data.x, data.edge_index, data.batch)
                loss = criterion(out.view(-1), data.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * data.num_graphs
                
            avg_loss = total_loss / len(loader.dataset)
            mlflow.log_metric("mse_loss", avg_loss)
            
            mlflow.pytorch.log_model(model, "model") # type: ignore
            print(f"Finished {run_name} with Loss: {avg_loss:.4f}")