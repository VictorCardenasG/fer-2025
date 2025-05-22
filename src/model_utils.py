import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torch.nn as nn

def calculate_metric(y, y_pred):
    return accuracy_score(y, y_pred)

def train_one_epoch(dataloader, model, optimizer, scheduler, cfg, criterion):
    model.train()
    final_y, final_y_pred, final_loss = [], [], []
    for X, y in tqdm(dataloader, desc="Training"):
        X, y = X.to(cfg["device"]), y.to(cfg["device"])
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        final_y.extend(y.cpu().numpy())
        final_y_pred.extend(y_pred.detach().cpu().numpy())
        final_loss.append(loss.item())

    metric = calculate_metric(final_y, np.argmax(final_y_pred, axis=1))
    return metric, np.mean(final_loss)

def validate_one_epoch(dataloader, model, cfg, criterion):
    model.eval()
    final_y, final_y_pred, final_loss = [], [], []
    for X, y in tqdm(dataloader, desc="Validating"):
        X, y = X.to(cfg["device"]), y.to(cfg["device"])
        with torch.no_grad():
            y_pred = model(X)
            loss = criterion(y_pred, y)

        final_y.extend(y.cpu().numpy())
        final_y_pred.extend(y_pred.cpu().numpy())
        final_loss.append(loss.item())

    metric = calculate_metric(final_y, np.argmax(final_y_pred, axis=1))
    return metric, np.mean(final_loss)

def fit(model, optimizer, scheduler, cfg, train_dl, val_dl, patience=3):
    acc_list, loss_list, val_acc_list, val_loss_list = [], [], [], []
    best_acc, counter = -1, 0
    criterion = nn.CrossEntropyLoss()

    for epoch in range(cfg["epochs"]):
        print(f"\nEpoch {epoch + 1}/{cfg['epochs']}")
        train_acc, train_loss = train_one_epoch(train_dl, model, optimizer, scheduler, cfg, criterion)
        val_acc, val_loss = validate_one_epoch(val_dl, model, cfg, criterion)

        acc_list.append(train_acc)
        loss_list.append(train_loss)
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)

        print(f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    return acc_list, loss_list, val_acc_list, val_loss_list, model
