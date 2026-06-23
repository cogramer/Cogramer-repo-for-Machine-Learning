CONFIG = {
    "input_size":    12,     # 6 weather + hour_sin + hour_cos + day_sin + day_cos + delta + mask
    "hidden_size":   128,
    "num_layers":    2,
    "output_size":   1,      # 1 = single-var, N = multi-var
    "dropout":       0.2,
    "batch_size":    16,
    "num_epochs":    100,
    "lr":            1e-3,
    "lr_patience":   10,     # epochs before LR halves
    "early_stop":    15,     # epochs before early stop
    "grad_clip":     1.0,
    "model_path":    "best_model.pth",
}