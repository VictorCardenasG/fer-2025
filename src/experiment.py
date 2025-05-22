import torch
import os
import timm
import mlflow
import mlflow.pytorch
from model_utils import fit

def run_experiment(cfg, train_dataset, valid_dataset, backbone, batch_size, learning_rate):
    cfg.update({"backbone": backbone, "batch_size": batch_size, "learning_rate": learning_rate})

    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dl = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    model = timm.create_model(backbone, pretrained=True, num_classes=cfg["n_classes"]).to(cfg["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(train_dl) * cfg["epochs"],
        eta_min=cfg["lr_min"]
    )

    with mlflow.start_run(run_name=f"{backbone}_bs{batch_size}_lr{learning_rate}"):
        mlflow.log_params(cfg)

        acc, loss, val_acc, val_loss, model = fit(model, optimizer, scheduler, cfg, train_dl, valid_dl)

        mlflow.log_metrics({
            "final_val_accuracy": val_acc[-1],
            "final_val_loss": val_loss[-1]
        })

        model_path = os.path.join("models", f"{backbone}_bs{batch_size}_lr{learning_rate}.pth")
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), model_path)
        mlflow.pytorch.log_model(model, "model")
        mlflow.log_artifact(model_path)
