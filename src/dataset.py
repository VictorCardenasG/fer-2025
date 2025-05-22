import os
import cv2
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

sub_folders = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
labels = list(range(len(sub_folders)))

def load_dataframe(root_dir):
    data = []
    for s, l in zip(sub_folders, labels):
        for r, d, f in os.walk(os.path.join(root_dir, s)):
            for file in f:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    data.append((os.path.join(s, file), l))
    return pd.DataFrame(data, columns=["file_name", "label"])

class CustomDataset(Dataset):
    def __init__(self, cfg, df, transform=None):
        self.root_dir = cfg["root_dir"]
        self.df = df
        self.file_names = df["file_name"].values
        self.labels = df["label"].values
        self.transform = transform or A.Compose([
            A.Resize(cfg["image_size"], cfg["image_size"]),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.labels[idx]
        file_path = os.path.join(self.root_dir, self.file_names[idx])
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=image)
        image = augmented["image"] / 255.0
        return image, label
