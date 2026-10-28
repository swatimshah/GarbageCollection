import numpy as np
import os
from pathlib import Path
from PIL import Image

def downsize_image(my_input_dir, my_output_dir, quality=85):
    """
    Resizes an image maintaining aspect ratio and compresses it.
    """
    count = 0

    for entry in os.listdir(my_input_dir):
        full_path = os.path.join(my_input_dir, entry)
        if os.path.isdir(full_path):
            print(full_path)

        p = Path(full_path)
        my_images = list(p.glob("*.JPG"))
        for my_input_file in my_images:
            print(f"Image {my_input_file}")
            with Image.open(my_input_file) as img:
                #print("**********************************")
                #print(img.size)
                #print(img.mode)
                #print("**********************************")
                resized_img = img.resize((240, 180))

                # Save with optimized quality
                # Not recommended for cross-platform compatibility

                my_output_file = my_output_dir + "/" + entry + "/" + str(my_input_file.stem) + "_resized"
                resized_img.save(my_output_file + ".JPEG", optimize=True, quality=quality)
                print(f"Image saved to: {my_output_file}")



# Example Usage on Redmi
# Ensure you have permission to access storage in Pydroid 3

input_dir = 'D:/TestData-Garbage-Collection/input'
output_dir = 'D:/TestData-Garbage-Collection/resized'

downsize_image(input_dir, output_dir)

#unseendata_input_dir1 = 'D:\GarbagePhotos\Test1\Input'
#unseendata_output_dir1 = 'D:\GarbagePhotos\Test1\Resized'

#downsize_image(unseendata_input_dir1, unseendata_output_dir1)

#unseendata_input_dir3 = 'D:\GarbagePhotos\Test3\Input'
#unseendata_output_dir3 = 'D:\GarbagePhotos\Test3\Resized'

#downsize_image(unseendata_input_dir3, unseendata_output_dir3)

#unseendata_input_dir4 = 'D:\GarbagePhotos\Test4\Input'
#unseendata_output_dir4 = 'D:\GarbagePhotos\Test4\Resized'

#downsize_image(unseendata_input_dir4, unseendata_output_dir4)

#unseendata_input_dir5 = 'D:\GarbagePhotos\Test5\Input'
#unseendata_output_dir5 = 'D:\GarbagePhotos\Test5\Resized'

#downsize_image(unseendata_input_dir5, unseendata_output_dir5)

incremental_training_input = 'D:/Incremental_Training_Data/Input2'
incremental_training_resized = 'D:/Incremental_Training_Data/Resized2'

downsize_image(incremental_training_input, incremental_training_resized)


