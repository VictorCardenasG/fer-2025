import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import os
import timm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt  
import seaborn as sns 
import random

# Define configuration (use the same configuration used for training)
cfg = {
    "image_size": 252,
    "n_classes": 6,
    "backbone": 'resnet18',
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
}

# Define Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, cfg, image_path, label=None, transform=None):
        self.image_path = image_path
        self.label = label
        self.transform = transform or A.Compose([
            A.Resize(cfg["image_size"], cfg["image_size"]),
            ToTensorV2(),
        ])
        
    def __len__(self):
        return len(self.image_path)  # Update to return the number of images
    
    def __getitem__(self, idx):
        file_path = os.path.normpath(self.image_path[idx])  # Normalize the file path
        
        try:
            image = cv2.imread(file_path)
            if image is None:
                raise FileNotFoundError(f"Failed to load image from: {file_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            augmented = self.transform(image=image)
            image = augmented['image'] / 255  # Normalize image to [0, 1]
            
            label = self.label[idx] if self.label is not None else None
            
            return image, label, file_path  # Return image, label, and file path
        except Exception as e:
            print(f"Error loading image: {e}")
            return None

# Load the saved model
model = timm.create_model(cfg["backbone"], pretrained=False, num_classes=cfg["n_classes"]).to(cfg["device"])
model_path = r"C:\Users\Victor Cardenas\Documents\msc\semestre-4\idi-4\fer-2025\models\model_notransforms_lr5e4_1000_6_emotions_res18_ai.h5"


# Load the model state
model.load_state_dict(torch.load(model_path, map_location=cfg["device"]))
model.eval()  # Set the model to evaluation mode

# Create directory for incorrect predictions if it doesn't exist
incorrect_preds_dir = "incorrect_predictions_2"
os.makedirs(incorrect_preds_dir, exist_ok=True)

# Function to evaluate the model on a validation set and save misclassified images
def evaluate_model(validation_folder):
    emotions = ["Anger", "Disgust", "Fear", "Happy", "Sad", "Surprise"]
    y_true = []
    y_pred = []
    image_paths = []
    
    # Counters for correct and incorrect predictions
    correct_predictions = {emotion: 0 for emotion in emotions}
    incorrect_predictions = {emotion: 0 for emotion in emotions}
    
    for emotion in emotions:
        folder_path = os.path.join(validation_folder, emotion)
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(folder_path, file_name)
                image_paths.append(image_path)
                y_true.append(emotions.index(emotion))  # True label as index
    
    # Create dataset and dataloader
    dataset = CustomDataset(cfg, image_paths, label=y_true)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Evaluate the model
    with torch.no_grad():
        for idx, (images, labels, paths) in enumerate(dataloader):
            if images is None:
                continue

            images = images.to(cfg["device"])
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            y_pred.extend(predicted.cpu().numpy())

            # Check for correct and incorrect predictions
            for true, pred, path in zip(labels.numpy(), predicted.cpu().numpy(), paths):
                if true != pred:
                    # Save incorrectly predicted image with the specified filename format
                    true_emotion = emotions[true]
                    pred_emotion = emotions[pred]
                    filename = f"img_{idx:02d}_true{true_emotion}_predicted{pred_emotion}.jpg"
                    save_path = os.path.join(incorrect_preds_dir, filename)

                    # Convert the image tensor to saveable format and save
                    save_image = (images.squeeze().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    cv2.imwrite(save_path, cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR))

                    print(f"Incorrect Prediction: True: {true_emotion}, Predicted: {pred_emotion}, Saved as: {filename}")
                    incorrect_predictions[true_emotion] += 1
                else:
                    correct_predictions[emotions[true]] += 1

    # Compute accuracy
    accuracy = np.mean(np.array(y_pred) == np.array(y_true))
    print(f"Accuracy: {accuracy:.2f}")

    # Confusion matrix and classification report
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, target_names=emotions)

    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=emotions, yticklabels=emotions)
    plt.xlabel('Predicted Emotion')
    plt.ylabel('True Emotion')
    plt.title('Confusion Matrix')
    plt.show()

    # Display random samples with true and predicted labels
    display_random_samples(dataset, y_pred, y_true)

    # Print total correct and incorrect predictions
    print("\nSummary of Predictions:")
    for emotion in emotions:
        total_predictions = correct_predictions[emotion] + incorrect_predictions[emotion]
        print(f"{emotion}: Correct: {correct_predictions[emotion]}, Incorrect: {incorrect_predictions[emotion]}, Total: {total_predictions}")

def display_random_samples(dataset, y_pred, y_true, n_samples=20):
    # Sample random indices
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    
    plt.figure(figsize=(15, 10))
    
    for i, idx in enumerate(indices):
        image, true_label, _ = dataset[idx]
        predicted_label = y_pred[idx]
        
        plt.subplot(4, 5, i + 1)
        plt.imshow(image.permute(1, 2, 0).numpy())
        plt.title(f'True: {["Anger", "Disgust", "Fear", "Happy", "Sad", "Surprise"][true_label]}\n'
                  f'Pred: {["Anger", "Disgust", "Fear", "Happy", "Sad", "Surprise"][predicted_label]}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage
validation_folder = r"C:\Users\Victor Cardenas\Documents\msc\semestre-4\idi-4\fer-2025\data\processed\validation"
evaluate_model(validation_folder)
