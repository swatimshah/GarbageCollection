from PIL import Image
import os
os.environ["TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS"] = "1"
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import numpy as np
from pathlib import Path
from tensorflow.keras.preprocessing import image
import random
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, AveragePooling2D, Dropout
from tensorflow.keras import layers, models
import tensorflow as tf
from keras.layers import Input, Dense, concatenate
from matplotlib import pyplot
from numpy.random import seed
from tensorflow.random import set_seed
from tensorflow.keras.optimizers import Adam
from keras.constraints import min_max_norm
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import ModelCheckpoint
from imblearn.over_sampling import SMOTE
import pickle
import cv2
from keras.regularizers import L2

os.add_dll_directory("D:/GarbageCollection/cudnn-windows-x86_64-8.9.7.29_cuda11-archive/cudnn-windows-x86_64-8.9.7.29_cuda11-archive/bin")
print(tf.config.list_physical_devices('GPU'))

# setting the seed
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)
tf.keras.utils.set_random_seed(42)



def count_files_os(directory_path):
    count = 0
    for root, dirs, files in os.walk(directory_path):
        count = len(files)
    return count

def createDataList(my_output_dir, y_expected, quality=85):
    """
    Resizes an image maintaining aspect ratio and compresses it.
    """
    count = 0
    data_list = []

    for entry in os.listdir(my_output_dir):
        full_path = os.path.join(my_output_dir, entry)
        if os.path.isdir(full_path):
            print(full_path)

        p = Path(full_path)
        my_images = list(p.glob("*.JPEG"))
        for my_input_file in my_images:
            #print(f"Image {my_input_file}")
            with Image.open(my_input_file) as img:
    
                try:
                    my_image = np.array(img)
                    # print(y_expected[count])
                    # Create a dictionary for each image and its label
                    image_data = {
                        'image_data': my_image,
                        'label': y_expected[count]
                    }
                    data_list.append(image_data)
                except IOError:
                    print(f"Error loading image {filename}")
               
                count += 1 

    return data_list


def createDataSets(my_output_dir, my_output_dir_inc):

    # 2. Lazy load the incremental dataset
    train_ds_2, val_ds_2 = tf.keras.utils.image_dataset_from_directory(
        my_output_dir_inc,
        validation_split=0.2,
        subset="both",
        seed=123,
        image_size=(180, 240),
        batch_size=10,
        label_mode="categorical"  # Yields one-hot encoded labels for 3 classes      
    )

    # 2. Lazy load the training dataset
    train_ds_1, val_ds_1 = tf.keras.utils.image_dataset_from_directory(
        my_output_dir,
        validation_split=0.2,
        subset="both",
        seed=123,
        image_size=(180, 240),
        batch_size=20,
        label_mode="categorical"  # Yields one-hot encoded labels for 3 classes      
    )


    train_ds = train_ds_1.concatenate(train_ds_2)
    val_ds = val_ds_1.concatenate(val_ds_2)

    return train_ds, val_ds




def train_model(train_ds, val_ds):

    # class_names = train_ds.class_names
    # print(f"Detected classes: {class_names}")
   
    # 4. Optimize dataset performance using caching and prefetching
    # AUTOTUNE dynamically adjusts background buffer sizes based on your CPU/GPU

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


    model = tf.keras.models.load_model(f'inc_model_checkpoints/ckpt_45.keras')


    model.compile(optimizer=Adam(learning_rate=0.000075),
        loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy'])

    model_checkpoint_callback = ModelCheckpoint(
        filepath='inc2_model_checkpoints/ckpt_{epoch:02d}.keras',
        save_freq='epoch'
    )

    # Train the model using the prepared data # 298, 261
    history = model.fit(train_ds, verbose=1, epochs=50, validation_data=val_ds, callbacks=[model_checkpoint_callback])


    # plot training and validation history
    pyplot.plot(history.history['loss'], label='tr_loss')
    pyplot.plot(history.history['val_loss'], label='val_loss')
    pyplot.plot(history.history['accuracy'], label='tr_accuracy')
    pyplot.plot(history.history['val_accuracy'], label='val_accuracy')
    pyplot.legend()
    pyplot.xlabel("No of iterations")
    pyplot.ylabel("Accuracy and loss")
    pyplot.show()

    return model;


def save_testdata(gc_model, test1_data, test3_data, test4_data, test5_data):

    with open('test1_data.pkl', 'wb') as file:
        pickle.dump(test1_data, file)
        
    with open('test3_data.pkl', 'wb') as file:
        pickle.dump(test3_data, file)

    with open('test4_data.pkl', 'wb') as file:
        pickle.dump(test4_data, file)

    with open('test5_data.pkl', 'wb') as file:
        pickle.dump(test5_data, file)
 

def generateDataList(output_dir):

    if os.path.exists(output_dir):

        num_files_leaves = count_files_os(output_dir + "\\leaves")
        print(f"Total files : {num_files_leaves}")
        leaves = [0] * num_files_leaves

        num_files_paper = count_files_os(output_dir + "\\paper")
        print(f"Total files : {num_files_paper}")
        paper = [1] * num_files_paper

        num_files_plastic = count_files_os(output_dir + "\\plastic")
        print(f"Total files : {num_files_plastic}")
        plastic = [2] * num_files_plastic

        result = np.concatenate((leaves, paper, plastic), axis=0)
        print(result)

        data = createDataList(output_dir, result)
    else:
        print("File not found.")
    
    return data




# Example Usage on Redmi
# Ensure you have permission to access storage in Pydroid 3
output_dir = 'D:/TestData-Garbage-Collection/resized'
inc_output_dir = 'D:/Incremental_Training_Data/Resized2'
unseendata_output_dir1 = 'D:/GarbagePhotos/Test1/resized'
unseendata_output_dir3 = 'D:/GarbagePhotos/Test3/Resized'
unseendata_output_dir4 = 'D:/GarbagePhotos/Test4/Resized'
unseendata_output_dir5 = 'D:/GarbagePhotos/Test5/Resized'



train_ds, val_ds = createDataSets(output_dir, inc_output_dir)
test1_data = generateDataList(unseendata_output_dir1)
test3_data = generateDataList(unseendata_output_dir3)
test4_data = generateDataList(unseendata_output_dir4)
test5_data = generateDataList(unseendata_output_dir5)
gc_model = train_model(train_ds, val_ds)
save_testdata(gc_model, test1_data, test3_data, test4_data, test5_data) 



