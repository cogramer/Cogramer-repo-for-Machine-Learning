import torch

def train_one_epoch(model, loader, loss_fn, optimizer, device, grad_clip):
    model.train()
    total_loss = 0.0
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        preds = model(batch_x)
        loss  = loss_fn(preds, batch_y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            preds = model(batch_x)
            total_loss += loss_fn(preds, batch_y).item()
    return total_loss / len(loader)


def train_model(model, train_loader, test_loader, loss_fn, optimizer, scheduler, cfg, device):
    train_hist, test_hist = [], []
    best_test_loss  = float("inf")
    patience_counter = 0

    for epoch in range(cfg["num_epochs"]):
        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, cfg["grad_clip"]
        )
        test_loss = evaluate(model, test_loader, loss_fn, device)

        train_hist.append(train_loss)
        test_hist.append(test_loss)
        scheduler.step(test_loss)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            torch.save(model.state_dict(), cfg["model_path"])
        else:
            patience_counter += 1
            if patience_counter >= cfg["early_stop"]:
                print(f"\nEarly stopping at epoch {epoch + 1} | Best test loss: {best_test_loss:.4f}")
                break

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{cfg['num_epochs']}] "
                f"Train: {train_loss:.4f}  Test: {test_loss:.4f}  "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

    return train_hist, test_hist