import os

# Set the path to your folder here
folder_path = r"C:\Users\Victor Cardenas\Documents\msc\semestre-4\idi-4\fer-2025\data\raw\april_gemini\fear"

# Valid image extensions (you can add more if needed)
image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

# List and sort all files with valid image extensions
images = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in image_extensions]
images.sort()  # Optional: ensures predictable ordering

# Rename images
for i, filename in enumerate(images):
    ext = os.path.splitext(filename)[1]
    new_name = f"fear_gemini_{i}{ext}"
    src = os.path.join(folder_path, filename)
    dst = os.path.join(folder_path, new_name)
    os.rename(src, dst)

print("Renaming complete.")