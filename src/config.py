import torch

cfg = {
    "root_dir": r"C:\Users\Victor Cardenas\Documents\msc\semestre-4\idi-4\fer-2025\data\processed\training_2025\train",
    "image_size": 256,
    "batch_size": 32,
    "n_classes": 7,
    "backbone": "resnet18",
    "learning_rate": 5e-4,
    "lr_min": 1e-6,
    "epochs": 25,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "seed": 42
}
