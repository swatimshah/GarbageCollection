import os
os.environ["TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS"] = "1"
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import tensorflow as tf
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
from numpy.random import seed
from tensorflow.random import set_seed
import random


# setting the seed
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)
tf.keras.utils.set_random_seed(42)


def test_model(test_data):

    data = [None] * len(test_data) 
    labels = [None] * len(test_data) 

    for i in range(len(test_data)):
        data[i] = test_data[i]['image_data']
        #print(data[i])

    for j in range(len(test_data)):
        labels[j] = test_data[j]['label']
        #print(labels[j])

    reshaped_user_data = np.array(data).reshape(len(data), 180, 240, 3)    
    reshaped_user_labels = np.array(labels).reshape(-1)  

    return reshaped_user_data, reshaped_user_labels

    #===================================================

def printAccuracy(user_data, user_labels):

    for i in range (1, 50, 1):
  
        if (i < 10):
            model = tf.keras.models.load_model(f'inc2_model_checkpoints/ckpt_0{i}.keras')
        else:
            model = tf.keras.models.load_model(f'inc2_model_checkpoints/ckpt_{i}.keras')


        with open("accuracy_details.csv", "a") as f:
            f.write(f"{i},")  

        for i in range (4):        

            predict_y = model.predict(user_data[i])
            Y_hat_classes = np.argmax(predict_y, axis=-1)
            confusionmatrix = confusion_matrix(user_labels[i], Y_hat_classes)
            accuracy = np.trace(confusionmatrix) / np.sum(confusionmatrix)
            percentage_acc = accuracy * 100
            with open("accuracy_details.csv", "a") as f:
                f.write(f"{percentage_acc:.2f},")  

        with open("accuracy_details.csv", "a") as f:
            f.write(f"\n")  
    

with open('test1_data.pkl', 'rb') as file:
    test1_data = pickle.load(file)

with open('test3_data.pkl', 'rb') as file:
    test3_data = pickle.load(file)

with open('test4_data.pkl', 'rb') as file:
    test4_data = pickle.load(file)

with open('test5_data.pkl', 'rb') as file:
    test5_data = pickle.load(file)

user_data = [None] * 4
user_labels = [None] * 4

user_data[0], user_labels[0] = test_model(test1_data)
user_data[1], user_labels[1] = test_model(test3_data)
user_data[2], user_labels[2] = test_model(test4_data)
user_data[3], user_labels[3] = test_model(test5_data)

printAccuracy(user_data, user_labels)