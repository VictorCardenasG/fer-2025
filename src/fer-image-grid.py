import os
import re
from PIL import Image

def split_faces_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.jpeg'):
            name_part = os.path.splitext(filename)[0]
            match = re.match(r"(\d{2})", name_part)
            if not match:
                print(f"Skipping {filename} — doesn't start with two digits.")
                continue

            rc = match.group(1)
            rows = int(rc[0])
            cols = int(rc[1])

            image_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(image_path)
            except Exception as e:
                print(f"Could not open {filename}: {e}")
                continue

            width, height = img.size
            cell_width = width // cols
            cell_height = height // rows

            face_count = 0
            for row in range(rows):
                for col in range(cols):
                    left = col * cell_width
                    upper = row * cell_height
                    right = left + cell_width
                    lower = upper + cell_height

                    face = img.crop((left, upper, right, lower))
                    face_filename = f"{name_part}_face_{row+1}_{col+1}.jpeg"
                    face.save(os.path.join(folder_path, face_filename))
                    face_count += 1

            print(f"{filename}: saved {face_count} face images in '{folder_path}'")

# Example usage:
split_faces_in_folder(r"C:\Users\Victor Cardenas\Documents\msc\semestre-4\idi-4\fer-2025\data\raw\march_gemini\angry_grid")
