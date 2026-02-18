from PIL import Image
import os
import numpy as np
from pathlib import Path
from tensorflow.keras.preprocessing import image

def downsize_image(my_input_dir, my_output_dir, quality=85):
    """
    Resizes an image maintaining aspect ratio and compresses it.
    """
    img_batch = np.empty((0, 300, 400, 3), dtype=float)

    for entry in os.listdir(my_input_dir):
        full_path = os.path.join(my_input_dir, entry)
        if os.path.isdir(full_path):
            print(full_path)

        p = Path(full_path)
        my_images = list(p.glob("*.JPG"))
        for my_input_file in my_images:
            print(f"Image {my_input_file}")
            with Image.open(my_input_file) as img:
                resized_img = img.resize((400, 300))

                # Save with optimized quality
                # Not recommended for cross-platform compatibility

                my_output_file = my_output_dir + "/" + entry + "/" + str(my_input_file.stem) + "_resized"
                resized_img.save(my_output_file + ".JPEG", optimize=True, quality=quality)
                print(f"Image saved to: {my_output_file}")

                # 1. Convert to array and Normalize (0-1)
                img_array = image.img_to_array(resized_img) / 255.0

                # 2. Add batch dimension: (300, 400, 3) -> (1, 300, 400, 3)
                temp = np.expand_dims(img_array, axis=0)
                print(temp.shape)
                img_batch = np.append(img_batch, temp, axis = 0)

                print(img_batch.shape)


# Example Usage on Redmi
# Ensure you have permission to access storage in Pydroid 3
input_dir = 'D:/TestData-Garbage-Collection/input'
output_dir = 'D:/TestData-Garbage-Collection/resized'


if os.path.exists(input_dir):
    downsize_image(input_dir, output_dir)
else:
    print("File not found.")