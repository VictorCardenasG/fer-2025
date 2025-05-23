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

root_dir = r"C:\Users\Victor Cardenas\Documents\msc\semestre-4\idi-4\fer-2025\data\processed\training_2025\train"
sub_folders = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
labels = [0, 1, 2, 3, 4, 5, 6]

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

cfg = {
    "root_dir": root_dir,
    "image_size": 256,
    "batch_size": 32,
    "n_classes": 7,
    "backbone": 'resnet18',  
    "learning_rate": 5e-4,
    "lr_min": 1e-6,
    "epochs": 30,
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
    weight_decay=1e-4  
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

def fit(model, optimizer, scheduler, cfg, train_dataloader, valid_dataloader=None, patience=3):
    acc_list = []
    loss_list = []
    val_acc_list = []
    val_loss_list = []

    best_val_acc = -1
    epochs_without_improvement = 0

    for epoch in range(cfg["epochs"]):
        print(f"Epoch {epoch + 1}/{cfg['epochs']}")
        set_seed(cfg["seed"] + epoch)

        acc, loss = train_one_epoch(train_dataloader, model, optimizer, scheduler, cfg)

        if valid_dataloader:
            val_acc, val_loss = validate_one_epoch(valid_dataloader, model, cfg)

        acc_list.append(acc)
        loss_list.append(loss)

        if valid_dataloader:
            val_acc_list.append(val_acc)
            val_loss_list.append(val_loss)
            print(f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_without_improvement = 0
                print(" Validation improved — resetting patience.")
                # Optionally save best model
                torch.save(model.state_dict(), f"best_model.pth")
            else:
                epochs_without_improvement += 1
                print(f" No improvement. Patience counter: {epochs_without_improvement}/{patience}")

                if epochs_without_improvement >= patience:
                    print(" Early stopping triggered.")
                    break

        print(f"Loss: {loss:.4f} Acc: {acc:.4f}")
        print(f"Saving model model_{epoch}.pth")
        torch.save(model.state_dict(), f"model_{epoch}.pth")

    return acc_list, loss_list, val_acc_list, val_loss_list, model

def visualize_history(acc, loss, val_acc, val_loss):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('black')  # Set background for the whole figure

    for i in range(2):
        ax[i].set_facecolor('black')       # Set background for each subplot
        ax[i].tick_params(colors='white')  # White ticks
        ax[i].spines['bottom'].set_color('white')
        ax[i].spines['top'].set_color('white') 
        ax[i].spines['left'].set_color('white') 
        ax[i].spines['right'].set_color('white') 
        ax[i].xaxis.label.set_color('white')
        ax[i].yaxis.label.set_color('white')
        ax[i].title.set_color('white')
        ax[i].legend(loc="upper right", facecolor='black', edgecolor='white', labelcolor='white')

    ax[0].plot(range(len(loss)), loss, color='lightgray', label='train')
    ax[0].plot(range(len(val_loss)), val_loss, color='skyblue', label='valid')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('Epochs')

    ax[1].plot(range(len(acc)), acc, color='lightgray', label='train')
    ax[1].plot(range(len(val_acc)), val_acc, color='skyblue', label='valid')
    ax[1].set_title('Accuracy')
    ax[1].set_xlabel('Epochs')

    plt.tight_layout()
    plt.show()


acc, loss, val_acc, val_loss, model = fit(
    model,
    optimizer,
    scheduler,
    cfg,
    train_dataloader,
    valid_dataloader,
    patience=3
)

visualize_history(acc, loss, val_acc, val_loss)

torch.save(model.state_dict(), r'C:\Users\Victor Cardenas\Documents\msc\semestre-4\idi-4\fer-2025\models\model_7emotions_28_april_2000imgs_res18_ai.pth')

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