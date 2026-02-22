from PIL import Image
import os
import numpy as np
from pathlib import Path
from tensorflow.keras.preprocessing import image
import random
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras import layers, models
import tensorflow as tf
from keras.layers import Input, Dense, concatenate
from matplotlib import pyplot
from numpy.random import seed
from tensorflow.random import set_seed
from tensorflow.keras.optimizers import Adam

# setting the seed
seed(1)
set_seed(1)

def count_files_os(directory_path):
    count = 0
    for root, dirs, files in os.walk(directory_path):
        count = len(files)
    return count


def downsize_image(my_input_dir, my_output_dir, y_train, quality=85):
    """
    Resizes an image maintaining aspect ratio and compresses it.
    """
    img_batch = np.empty((0, 200, 150, 3), dtype=float)
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
                resized_img = img.resize((200, 150))

                # Save with optimized quality
                # Not recommended for cross-platform compatibility

                my_output_file = my_output_dir + "/" + entry + "/" + str(my_input_file.stem) + "_resized"
                resized_img.save(my_output_file + ".JPEG", optimize=True, quality=quality)
                print(f"Image saved to: {my_output_file}")

                # 1. Convert to array and Normalize (0-1)
                img_array = image.img_to_array(resized_img) / 255.0
                print(img_array.shape)
                
                try:
                    my_image = np.array(img_array)
                    print(result[count])
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

    # Randomize the input
    seed_value = 42

    # Set the seed before shuffling
    random.seed(seed_value) 

    # Shuffle the list in-place
    random.shuffle(training_data_list)	

    # divide training and test set
    train_list, test_list = train_test_split(training_data_list, test_size=0.3, random_state=42)

    data = [None] * len(train_list) 
    labels = [None] * len(train_list) 
    test_data = [None] * len(test_list) 
    test_label = [None] * len(test_list) 	

    for i in range(len(train_list)):
        data[i] = train_list[i]['image_data']
        print(data[i])

    for j in range(len(train_list)):
        labels[j] = np.array(train_list[j]['label']).reshape(-1)
        print(labels[j])

    for k in range(len(test_list)):
        test_data[k] = test_list[k]['image_data']
        print(test_data[k])

    for l in range(len(test_list)):
        test_label[l] = np.array(test_list[l]['label']).reshape(-1)
        print(test_label[l])

    reshaped_data = np.array(data).reshape(len(data), 200, 150, 3)    
    reshaped_labels = np.array(labels).reshape(len(labels))    
    reshaped_test_data = np.array(test_data).reshape(len(test_data), 200, 150, 3)    
    reshaped_test_labels = np.array(test_label).reshape(len(test_label))    


    # train the cnn
    model = Sequential()

    # Add Convolutional and Pooling layers for feature extraction
    model.add(Conv2D(48, (5, 5), activation='relu', input_shape=(200, 150, 3)))
    model.add(MaxPooling2D((3, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((3, 3)))
    # Flatten the 3D feature maps to a 1D vector for the fully connected layers
    model.add(GlobalAveragePooling2D())
    # Add Fully Connected (Dense) layers for classification
    model.add(Dense(40, activation='relu'))
    model.add(Dense(3, activation='softmax')) # Output layer for 10 classes

    # View the model summary
    model.summary()

    model.compile(optimizer=Adam(learning_rate=0.0004),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'])

    # Train the model using the prepared data
    history = model.fit(reshaped_data, reshaped_labels, verbose=1, epochs=600, validation_data=(reshaped_test_data, reshaped_test_labels), batch_size=5)


    # plot training and validation history
    pyplot.plot(history.history['loss'], label='tr_loss')
    pyplot.plot(history.history['val_loss'], label='val_loss')
    pyplot.plot(history.history['accuracy'], label='tr_accuracy')
    pyplot.plot(history.history['val_accuracy'], label='val_accuracy')
    pyplot.legend()
    pyplot.xlabel("No of iterations")
    pyplot.ylabel("Accuracy and loss")
    pyplot.show()


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