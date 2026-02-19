from PIL import Image
import os
import numpy as np
from pathlib import Path
from tensorflow.keras.preprocessing import image

def count_files_os(directory_path):
    count = 0
    for root, dirs, files in os.walk(directory_path):
        count = len(files)
    return count


def downsize_image(my_input_dir, my_output_dir, y_train, quality=85):
    """
    Resizes an image maintaining aspect ratio and compresses it.
    """
    img_batch = np.empty((0, 300, 400, 3), dtype=float)
    count = 0
    training_data_list = []

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
                img_exp = np.expand_dims(img_array, axis=0)
                print(img_exp.shape)
                
                try:
                    my_image = np.array(img_exp)
                    # Create a dictionary for each image and its label
                    image_data = {
                        'image_data': my_image,
                        'label': result[count]
                    }
                    training_data_list.append(image_data)
                except IOError:
                    print(f"Error loading image {filename}")
               
                count += 1 

    print(training_data_list[50]['label'])	


# Example Usage on Redmi
# Ensure you have permission to access storage in Pydroid 3
input_dir = 'D:/TestData-Garbage-Collection/input'
output_dir = 'D:/TestData-Garbage-Collection/resized'

num_files_leaves = count_files_os(input_dir + "\\leaves")
print(f"Total files : {num_files_leaves}")
leaves = [0] * num_files_leaves

num_files_paper = count_files_os(input_dir + "\\paper")
print(f"Total files : {num_files_paper}")
paper = [1] * num_files_paper

num_files_plastic = count_files_os(input_dir + "\\plastic")
print(f"Total files : {num_files_plastic}")
plastic = [2] * num_files_plastic

result = np.concatenate((leaves, paper, plastic), axis=0)
print(result)


if os.path.exists(input_dir):
    print("Start ...")
    downsize_image(input_dir, output_dir, result)
else:
    print("File not found.")