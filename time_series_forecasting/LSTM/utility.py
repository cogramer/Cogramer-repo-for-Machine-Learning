import torch
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import numpy as np
import matplotlib.pyplot as plt

def run_inference(model, loader, device):
    """Returns (all_preds, all_targets, last_input_step0) as numpy arrays."""
    model.eval()
    all_preds, all_targets, all_inputs_last = [], [], []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            preds   = model(batch_x).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(batch_y.numpy())
            # last timestep, feature 0 — used for persistence baseline
            all_inputs_last.extend(batch_x[:, -1, 0].cpu().numpy())

    return (
        np.array(all_preds),
        np.array(all_targets),
        np.array(all_inputs_last).reshape(-1, 1),
    )


# ─────────────────────────────────────────────
# INVERSE TRANSFORM — swap these out for multi-var
# ─────────────────────────────────────────────
def inverse_single_var(scaled_vals, scaler, n_weather_features=6):
    """Inverse-transform a single target column (col 0 = temperature)."""
    pad_cols = n_weather_features - 1
    padded   = np.concatenate([scaled_vals, np.zeros((len(scaled_vals), pad_cols))], axis=1)
    return scaler.inverse_transform(padded)[:, 0:1]


def inverse_multi_var(scaled_vals, scaler):
    """
    Inverse-transform multiple target columns.
    scaled_vals shape: (N, n_targets) — must match scaler's feature count.
    Extend or pad as needed for your specific targets.
    """
    return scaler.inverse_transform(scaled_vals)


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_persist):
    mae_model       = mean_absolute_error(y_true, y_pred)
    mae_persistence = mean_absolute_error(y_true, y_persist)
    max_err         = np.max(np.abs(y_true - y_pred))
    mape            = mean_absolute_percentage_error(y_true, y_pred) * 100
    improvement     = (mae_persistence - mae_model) / mae_persistence * 100

    print("--- Extended Evaluation ---")
    print(f"MAE (real units):        {mae_model:.2f}°")
    print(f"Max single error:        {max_err:.2f}°")
    print(f"MAPE:                    {mape:.2f}%")
    print()
    print("--- Persistence Baseline Comparison ---")
    print(f"Naive baseline MAE:      {mae_persistence:.2f}°")
    print(f"Your model MAE:          {mae_model:.2f}°")
    print(f"Improvement over naive:  {improvement:.1f}%")


# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────
def plot_loss(train_hist, test_hist):
    epochs = np.arange(1, len(train_hist) + 1)
    plt.plot(epochs, train_hist, label="Training loss")
    plt.plot(epochs, test_hist,  label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Test Loss")
    plt.legend()
    plt.show()