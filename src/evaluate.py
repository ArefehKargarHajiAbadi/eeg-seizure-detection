import torch

def evaluate_dl(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            pred_label = (pred > 0.5).float()
            correct += (pred_label == y_batch).sum().item()
            total += y_batch.size(0)
    return correct / total