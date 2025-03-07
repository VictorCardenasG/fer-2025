# classifier.py
# venv used -> deep-learning-cuda
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import cv2

import os
os.environ['LIBPNG_WARNINGS'] = '0'

import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import os
import pandas as pd 

root_dir = r"C:\Users\Victor Cardenas\Documents\msc\semestre-4\idi-4\fer-2025\data\processed\train"
sub_folders = ["Anger", "Disgust", "Fear", "Happy", "Sad", "Surprise"]
labels = [0, 1, 2, 3, 4, 5]

data = []

for s, l in zip(sub_folders, labels):
    for r, d, f in os.walk(os.path.join(root_dir, s)):
        for file in f:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                data.append((os.path.join(s, file), l))

df = pd.DataFrame(data, columns=['file_name','label'])

# print(df)

from sklearn.model_selection import train_test_split

import timm

# Cambia la configuración para utilizar ResNet-18
cfg = {
    "root_dir": root_dir,
    "image_size": 256,
    "batch_size": 32,
    "n_classes": 6,
    "backbone": 'resnet18',  
    "learning_rate": 5e-4,
    "lr_min": 1e-6,
    "epochs": 2,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "seed": 42
}

print(cfg['device'])

class CustomDataset(Dataset):
    def __init__(self, cfg, df, transform=None):
        self.root_dir = cfg["root_dir"]
        self.df = df
        self.file_names = df['file_name'].values
        self.labels = df['label'].values

        if transform:
            self.transform = transform
        else:
            self.transform = A.Compose([
                              A.Resize(cfg["image_size"], cfg["image_size"]),
                              ToTensorV2(),
                           ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.labels[idx]
        file_path = os.path.join(self.root_dir, self.file_names[idx])

        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        augmented = self.transform(image=image)
        image = augmented['image']

        image = image / 255.0

        return image, label

example_dataset = CustomDataset(cfg, df)

example_dataloader = DataLoader(
    example_dataset,
    batch_size=cfg["batch_size"],
    shuffle=True,
    num_workers=0
)

X = df
y = df.label

train_df, valid_df, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply the custom transform for the training dataset
train_transform = A.Compose([
    A.Resize(cfg["image_size"], cfg["image_size"]),
    # A.Rotate(p=0.6, limit=[-45, 45]),
    # A.HorizontalFlip(p=0.6),
    # A.CoarseDropout(max_holes=1, max_height=64, max_width=64, p=0.3),
    # A.RandomBrightnessContrast(),  
    ToTensorV2()
])

# Keep a simple transformation for the validation dataset
valid_transform = A.Compose([
    A.Resize(cfg["image_size"], cfg["image_size"]),
    ToTensorV2()
])

# Apply the appropriate transform when creating the datasets
train_dataset = CustomDataset(cfg, train_df, transform=train_transform)
valid_dataset = CustomDataset(cfg, valid_df, transform=valid_transform)

# Dataloaders
train_dataloader = DataLoader(train_dataset,
                              batch_size=cfg["batch_size"],
                              shuffle=True)

valid_dataloader = DataLoader(valid_dataset,
                              batch_size=cfg["batch_size"],
                              shuffle=False)

# Cargar el modelo ResNet-18 preentrenado
model = timm.create_model(cfg["backbone"], pretrained=True, num_classes=cfg["n_classes"]).to(cfg["device"])

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=cfg["learning_rate"],
    weight_decay=1e-4  # You can experiment with values like 1e-4, 1e-5
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=np.ceil(len(train_dataloader.dataset) / cfg["batch_size"]) * cfg["epochs"],
    eta_min=cfg["lr_min"]
)

from sklearn.metrics import accuracy_score

def calculate_metric(y, y_pred):
    metric = accuracy_score(y, y_pred)
    return metric

def train_one_epoch(dataloader, model, optimizer, scheduler, cfg):
    model.train()
    final_y = []
    final_y_pred = []
    final_loss = []

    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        X = batch[0].to(cfg["device"])
        y = batch[1].to(cfg["device"])

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            y_pred = model(X)
            loss = criterion(y_pred, y)

            y = y.detach().cpu().numpy().tolist()
            y_pred = y_pred.detach().cpu().numpy().tolist()

            final_y.extend(y)
            final_y_pred.extend(y_pred)
            final_loss.append(loss.item())

            loss.backward()
            optimizer.step()

        scheduler.step()

    loss = np.mean(final_loss)
    final_y_pred = np.argmax(final_y_pred, axis=1)
    metric = calculate_metric(final_y, final_y_pred)

    return metric, loss

def validate_one_epoch(dataloader, model, cfg):
    model.eval()
    final_y = []
    final_y_pred = []
    final_loss = []

    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        X = batch[0].to(cfg["device"])
        y = batch[1].to(cfg["device"])

        with torch.no_grad():
            y_pred = model(X)
            loss = criterion(y_pred, y)

            y = y.detach().cpu().numpy().tolist()
            y_pred = y_pred.detach().cpu().numpy().tolist()

            final_y.extend(y)
            final_y_pred.extend(y_pred)
            final_loss.append(loss.item())

    loss = np.mean(final_loss)
    final_y_pred = np.argmax(final_y_pred, axis=1)
    metric = calculate_metric(final_y, final_y_pred)

    return metric, loss


import random

def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(cfg["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

cfg["seed"] = 42

def fit(model, optimizer, scheduler, cfg, train_dataloader, valid_dataloader=None):
    acc_list = []
    loss_list = []
    val_acc_list = []
    val_loss_list = []

    for epoch in range(cfg["epochs"]):
        print(f"Epoch {epoch + 1}/{cfg['epochs']}")

        set_seed(cfg["seed"] + epoch)

        acc, loss = train_one_epoch(train_dataloader, model, optimizer, scheduler, cfg)

        if valid_dataloader:
            val_acc, val_loss = validate_one_epoch(valid_dataloader, model, cfg)

        print(f'Loss: {loss:.4f} Acc: {acc:.4f}')
        acc_list.append(acc)
        loss_list.append(loss)

        if valid_dataloader:
            print(f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')
            val_acc_list.append(val_acc)
            val_loss_list.append(val_loss)

        print(f'Saving model model_{epoch}.h5')
        torch.save(model.state_dict(), f'model_{epoch}.h5')

    return acc_list, loss_list, val_acc_list, val_loss_list, model

def visualize_history(acc, loss, val_acc, val_loss):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(range(len(loss)), loss, color='darkgrey', label='train')
    ax[0].plot(range(len(val_loss)), val_loss, color='cornflowerblue', label='valid')
    ax[0].set_title('Loss')

    ax[1].plot(range(len(acc)), acc, color='darkgrey', label='train')
    ax[1].plot(range(len(val_acc)), val_acc, color='cornflowerblue', label='valid')
    ax[1].set_title('Metric (Accuracy)')

    for i in range(2):
        ax[i].set_xlabel('Epochs')
        ax[i].legend(loc="upper right")
    plt.show()

acc, loss, val_acc, val_loss, model = fit(model, optimizer, scheduler, cfg, train_dataloader, valid_dataloader)

visualize_history(acc, loss, val_acc, val_loss)

torch.save(model.state_dict(), r'C:\Users\Victor Cardenas\Documents\msc\semestre-4\idi-4\fer-2025\models\model_vicfiltered_notransforms_lr5e4_1000_6_emotions_res18_ai.pth')

model.eval()

final_y = []
final_y_pred = []

for step, batch in tqdm(enumerate(valid_dataloader), total=len(valid_dataloader)):
    X = batch[0].to(cfg["device"])
    y = batch[1].to(cfg["device"])

    with torch.no_grad():
        y_pred = model(X)

        y = y.detach().cpu().numpy().tolist()
        y_pred = y_pred.detach().cpu().numpy().tolist()

        final_y.extend(y)
        final_y_pred.extend(y_pred)

final_y_pred = np.argmax(final_y_pred, axis=1)

metric = calculate_metric(final_y, final_y_pred)
print(f'final_metric: {metric}')