import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import torch
def train_hybrid_model(model, loader, optimizer, device, epoch):
    """Trains the hybrid model for one epoch.

    Args:
        model: The HybridModel instance.
        loader: The DataLoader providing batches of molecular data.
        optimizer: The optimizer used to update model weights
        device: The device (CPU, GPU or CUDA) where computations occurs.
    """
    model.train()
    criterion = torch.nn.MSELoss()
    total_loss = 0

    all_preds = []
    all_targets = []

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data.x, data.edge_index, data.batch, data.extra_features)
        loss = criterion(out.view(-1), data.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * data.num_graphs

        all_preds.append(out.detach().cpu().numpy().flatten())
        all_targets.append(data.y.detach().cpu().numpy().flatten())

    avg_mse = total_loss / len(loader.dataset)
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    avg_mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)

    print(
        f"Epoch {epoch} | "
        f"TRAIN_MSE={avg_mse:.4f} | "
        f"TRAIN_MAE={avg_mae:.4f} | "
        f"TRAIN_R2={r2:.4f}"
    )

    return avg_mse


def evaluate_hybrid_model(model, loader, device):
    """Evaluates hybrid model

    Args:
        model: The HybridModel instance.
        loader: The DataLoader providing batches of molecular data.
        device: The device (CPU, GPU or CUDA) where computations occurs.
    """
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch, data.extra_features)
            
            all_preds.append(out.detach().cpu().numpy().flatten())
            all_targets.append(data.y.detach().cpu().numpy().flatten())
            
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    
    return r2_score(targets, preds), mean_absolute_error(targets, preds), targets, preds

