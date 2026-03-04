from LSTM import LSTMmodel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def build_model(cfg, device):
    model = LSTMmodel(
        cfg["input_size"],
        cfg["hidden_size"],
        cfg["num_layers"],
        cfg["output_size"],
        cfg["dropout"],
    ).to(device)
    loss_fn  = nn.HuberLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=cfg["lr_patience"]
    )
    return model, loss_fn, optimizer, scheduler


def build_loaders(x_train, y_train, x_test, y_test, batch_size):
    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader