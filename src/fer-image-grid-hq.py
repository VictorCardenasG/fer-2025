import os
import re
from PIL import Image

def split_all_grids_in_folder(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".jpeg"):
                match = re.match(r"(\d{2})", file)
                if match:
                    grid_code = match.group(1)
                    rows = int(grid_code[0])
                    cols = int(grid_code[1])
                    image_path = os.path.join(root, file)
                    split_grid_image(image_path, rows, cols)

def split_grid_image(image_path, rows, cols):
    img = Image.open(image_path)
    width, height = img.size

    tile_width = 256
    tile_height = 256

    expected_width = cols * tile_width
    expected_height = rows * tile_height

    if width < expected_width or height < expected_height:
        print(f"Skipping {image_path} — image is too small for expected grid {rows}x{cols}")
        return

    base_dir = os.path.dirname(image_path)
    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    for row in range(rows):
        for col in range(cols):
            left = col * tile_width
            upper = row * tile_height
            right = left + tile_width
            lower = upper + tile_height

            tile = img.crop((left, upper, right, lower))
            tile_filename = f"{row+1:02d}{col+1:02d}.jpeg"
            tile.save(os.path.join(base_dir, tile_filename))

    print(f"Processed {image_path} → {rows * cols} images")

split_all_grids_in_folder(r"C:\Users\Victor Cardenas\Documents\msc\semestre-4\idi-4\fer-2025\data\raw\april_stable_diff\Neutral")
