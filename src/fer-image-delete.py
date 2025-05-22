
import os

# Set the path to your folder here
folder_path = r"C:\Users\Victor Cardenas\Documents\msc\semestre-4\idi-4\fer-2025\data\raw\april_gemini\fear"

# Image file extensions you want to target
image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.startswith("Fear") and os.path.splitext(filename)[1].lower() in image_extensions:
        file_path = os.path.join(folder_path, filename)
        os.remove(file_path)
        print(f"Deleted: {filename}")

print("Deletion complete.")
