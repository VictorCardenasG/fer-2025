import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
import numpy as np
from tqdm import tqdm

# Configuration
cfg = {
    "image_size": 252,
    "batch_size": 32,
    "n_classes": 7,  # Number of emotions
    "backbone": 'resnet18',
    "learning_rate": 5e-4,  # You can reduce this for fine-tuning (e.g., 1e-4 or 1e-5)
    "epochs": 5,  # Fine-tune for fewer epochs
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# Load the existing model
model_path = r"C:\Users\Victor Cardenas\Documents\msc\semestre-4\idi-4\fer-2025\models\model_notransforms_lr5e4_1000_6_emotions_res18_ai.pth"
model = timm.create_model(cfg["backbone"], pretrained=False, num_classes=cfg["n_classes"])
model.load_state_dict(torch.load(model_path, map_location=cfg["device"]))  # Load trained weights
model.to(cfg["device"])

# OPTIONAL: Freeze earlier layers to prevent them from being retrained
for param in model.parameters():
    param.requires_grad = False  # Freeze all layers
for param in model.get_classifier().parameters():  
    param.requires_grad = True  # Unfreeze only the last layer

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"])

# Define the dataset class (same as before)
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or A.Compose([
            A.Resize(cfg["image_size"], cfg["image_size"]),
            ToTensorV2(),
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        file_path = self.image_paths[idx]
        label = self.labels[idx]

        image = cv2.imread(file_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {file_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        augmented = self.transform(image=image)
        image = augmented["image"] / 255.0

        return image, label

# Load the new dataset (2,000 images)
new_images_folder = r"C:\path_to_new_images"
emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise"]
image_paths = []
labels = []

for emotion_idx, emotion in enumerate(emotions):
    emotion_folder = os.path.join(new_images_folder, emotion)
    for file_name in os.listdir(emotion_folder):
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(os.path.join(emotion_folder, file_name))
            labels.append(emotion_idx)

# Create DataLoader for new data
new_dataset = CustomDataset(image_paths, labels)
new_dataloader = DataLoader(new_dataset, batch_size=cfg["batch_size"], shuffle=True)

# Training Loop
model.train()  # Set model to training mode

for epoch in range(cfg["epochs"]):
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(new_dataloader, desc=f"Epoch {epoch+1}/{cfg['epochs']}"):
        images, labels = images.to(cfg["device"]), labels.to(cfg["device"])

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total * 100
    print(f"Epoch {epoch+1}/{cfg['epochs']} - Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")

# Save the updated model
updated_model_path = r"C:\Users\Victor Cardenas\Documents\msc\semestre-4\idi-4\fer-2025\models\model_updated_with_2000_images.pth"
torch.save(model.state_dict(), updated_model_path)

print(f"Updated model saved to {updated_model_path}")
