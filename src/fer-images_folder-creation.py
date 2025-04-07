import os
import random
import shutil
from pathlib import Path

# Set base paths
source_base = Path(r"C:\Users\Victor Cardenas\Documents\msc\semestre-4\idi-4\fer-2025\data\raw\march_gemini")
target_base = Path(r"C:\Users\Victor Cardenas\Documents\msc\semestre-4\idi-4\fer-2025\data\training_2025")

# Emotion folders
emotions = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Number of images to sample per emotion
num_images_per_class = 2140
split_ratio = 0.9  # 90% train, 10% validation

# Ensure base target directories exist
for emotion in emotions:
    (target_base / "train" / emotion).mkdir(parents=True, exist_ok=True)
    (target_base / "validation" / emotion).mkdir(parents=True, exist_ok=True)

for emotion in emotions:
    source_folder = source_base / emotion
    all_images = list(source_folder.glob("*.*"))
    
    if len(all_images) < num_images_per_class:
        raise ValueError(f"Not enough images in {emotion}. Found {len(all_images)}, need {num_images_per_class}")

    # Randomly sample 2140 images
    selected_images = random.sample(all_images, num_images_per_class)

    # Shuffle for randomness
    random.shuffle(selected_images)
    split_idx = int(num_images_per_class * split_ratio)
    train_images = selected_images[:split_idx]
    val_images = selected_images[split_idx:]

    # Copy train images
    for idx, image_path in enumerate(train_images, 1):
        new_filename = f"{emotion}_{idx}.jpg"
        target_path = target_base / "train" / emotion / new_filename
        shutil.copy(image_path, target_path)

    # Copy validation images
    for idx, image_path in enumerate(val_images, 1):
        new_filename = f"{emotion}_{split_idx + idx}.jpg"
        target_path = target_base / "validation" / emotion / new_filename
        shutil.copy(image_path, target_path)

print("✅ All images copied and renamed successfully.")
