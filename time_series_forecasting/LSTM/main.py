import torch
from model_setup import build_model, build_loaders
from preprocess_data import preprocess_data
from CONFIG import CONFIG
from training import train_model
from utility import plot_loss, run_inference, inverse_single_var, compute_metrics

if __name__ == "__main__":
    x_train, y_train, x_test, y_test, scaler = preprocess_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}\n")

    model, loss_fn, optimizer, scheduler = build_model(CONFIG, device)
    train_loader, test_loader = build_loaders(
        x_train, y_train, x_test, y_test, CONFIG["batch_size"]
    )

    train_hist, test_hist = train_model(
        model, train_loader, test_loader, loss_fn, optimizer, scheduler, CONFIG, device
    )
    plot_loss(train_hist, test_hist)

    # Load best weights
    model.load_state_dict(torch.load(CONFIG["model_path"], weights_only=True))
    print("\nLoaded best model weights for inference.")

    all_preds, all_targets, all_inputs_last = run_inference(model, test_loader, device)

    # ── Swap inverse_single_var ↔ inverse_multi_var here ──
    y_pred_real    = inverse_single_var(all_preds,       scaler)
    y_true_real    = inverse_single_var(all_targets,     scaler)
    y_persist_real = inverse_single_var(all_inputs_last, scaler)

    compute_metrics(y_true_real, y_pred_real, y_persist_real)