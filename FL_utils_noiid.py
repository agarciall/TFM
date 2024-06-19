import os
import pickle
import psutil
import time
import threading
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from datetime import datetime
from tensorflow.keras.applications import ResNet50
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.utils import class_weight
import random
import shutil
import copy
import gc 
from keras import backend as K 
import sys
import pandas as pd
from PIL import Image


from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


def log_memory_and_time():
    process = psutil.Process(os.getpid())
    memory = process.memory_info().rss / 1024 ** 2
    print(f"Memory usage: {memory} MB")
    print(f"Time: {time.time()}")
    return memory



class MemoryUsageCallback(Callback):
    def __init__(self):
        super().__init__()
        self.previous_memory = 0
        self.memory_difference = 0

    def on_epoch_end(self, epoch, logs=None):
        process = psutil.Process(os.getpid())
        current_memory = process.memory_info().rss / 1024 ** 2
        self.memory_difference = current_memory - self.previous_memory
        self.previous_memory = current_memory

    
    

def save_model_and_history(model, history, directory):
    model.save(os.path.join(directory, 'model.h5'))
    with open(os.path.join(directory, 'history.pickle'), 'wb') as f:
        pickle.dump(history, f)



from tensorflow.keras import regularizers

def create_model():
    baseModel = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    headModel = baseModel.output
    headModel = layers.AveragePooling2D(pool_size=(4, 4))(headModel)
    headModel = layers.Flatten(name="flatten")(headModel)
    headModel = layers.Dense(64, activation="relu")(headModel) #, kernel_regularizer=regularizers.l2(0.01))(headModel)
    headModel = layers.Dropout(0.5)(headModel)
    headModel = layers.Dense(9, activation="softmax")(headModel) 

    model = models.Model(inputs=baseModel.input, outputs=headModel)

    total_layers = len(model.layers)

    trainable_layers = int(0.3 * total_layers)

    # Set the last 30% of layers as trainable and get their indices
    trainable_layer_indices = []
    weight_layer_indices = []
    for i in range(-trainable_layers, 0):
        model.layers[i].trainable = True
        trainable_layer_indices.append(i)
        if model.layers[i].weights:
            weight_layer_indices.append(i)

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model, trainable_layer_indices
    


def add_dummy_image_to_empty_classes(directory, dummy_image_path):
    for class_dir in os.listdir(directory):
        if not os.listdir(os.path.join(directory, class_dir)): 
            shutil.copy(dummy_image_path, os.path.join(directory, class_dir))

    
def get_data_for_round(client_number, round_number):

    class_indices = {
        "Basophils": 0,
        "Eosinophils": 1,
        "Erythroblasts": 2,
        "IG": 3,
        "Lymphoblasts": 4,
        "Lymphocytes": 5,
        "Monocytes": 6,
        "Neutrophils": 7,
        "Platelet": 8
    }
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale            = 1/255,  
    featurewise_center   = False,
    samplewise_center    = False,
    featurewise_std_normalization = False,
    samplewise_std_normalization  = False,
    zca_whitening                 = False,

    rotation_range     = 20,
    horizontal_flip    = True,
    vertical_flip      = True,

    width_shift_range  = 0.1, 
    height_shift_range = 0.1,
    shear_range        = 0.01, 
    zoom_range         = [0.4, 1.6], 
    data_format        = "channels_last",
    brightness_range   = [0.2, 1.0],
    #class_mode = "categorical"
    )

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale= 1/255)


    train_data = datagen.flow_from_directory(f'FL_noiid_split_40/train/client_{client_number}/part{round_number}', batch_size=8, seed = 66, target_size=(224, 224), interpolation='bilinear', shuffle=True, classes = class_indices)
    validation_data = datagen.flow_from_directory(f'FL_noiid_split_40/validation/client_{client_number}/part{round_number}', batch_size = 1, seed = 66, target_size=(224, 224), interpolation='bilinear', shuffle=False, classes = class_indices)

    
    class_weights = class_weight.compute_class_weight('balanced', np.unique(train_data.classes), train_data.classes)
    class_weights = dict(enumerate(class_weights))


    return train_data, validation_data, class_weights


def train_model(model, train_data, validation_data, trainable_layer_indices, round_number, client_number, memory_df, class_weight, epochs):
    memory_callback = MemoryUsageCallback()
    history = model.fit(
        train_data,
        steps_per_epoch=len(train_data),
        validation_data=validation_data,
        validation_steps=len(validation_data),
        epochs=epochs,
        callbacks=[memory_callback],
        class_weight=class_weight
    )
    
    weights = [layer.get_weights() for layer in model.layers if layer.name in trainable_layer_indices]

    memory_usage_df = memory_df

    memory_usage_df.loc[len(memory_usage_df)] = [round_number, client_number, memory_callback.memory_difference]

    return history, weights, model





def evaluate_model_on_clients(model, validation_generators):
    total_loss = 0
    total_accuracy = 0

    for generator in validation_generators:
        loss, accuracy = model.evaluate(generator)
        total_loss += loss
        total_accuracy += accuracy

    average_loss = total_loss / len(validation_generators)
    average_accuracy = total_accuracy / len(validation_generators)

    return average_loss, average_accuracy




from tensorflow.keras.utils import to_categorical

def generate_validation_images(num_rounds, num_clients):
    validation_labels = [[] for _ in range(num_clients)]  # Initialize a list of empty lists for labels

    label_map = {
    "Basophils": 0,
    "Eosinophils": 1,
    "Erythroblasts": 2,
    "IG": 3,
    "Lymphoblasts": 4,
    "Lymphocytes": 5,
    "Monocytes": 6,
    "Neutrophils": 7,
    "Platelet": 8
    }
    num_classes = len(label_map)
    for round_number in range(num_rounds):
        for client_number in range(num_clients):
            # Load validation data for this round and client
            validation_dir = f'FL_noiid_split_40/validation/client_{client_number}/part{round_number+1}'
            for class_dir in os.listdir(validation_dir):
                class_dir_path = os.path.join(validation_dir, class_dir)
                if os.path.isdir(class_dir_path):
                    for filename in os.listdir(class_dir_path):
                        img_path = os.path.join(class_dir_path, filename)
                        if os.path.isfile(img_path):
                            img = load_img(img_path, target_size=(224, 224))  
                            img = img_to_array(img) 
                            img = img/255.0

                            # Append the image array to the list for this client
                            validation_images[client_number].append(img)

                            # Get the label from the filename and append it to the list for this client
                            label = os.path.basename(os.path.dirname(img_path))  
                            #label_encoded = to_categorical(label_map[label], num_classes=num_classes)
                            validation_labels[client_number].append(label)

    return validation_images, validation_labels



def add_validation_images_for_round(validation_images, validation_labels, round_number, num_clients):

    label_map = {
    "Basophils": 0,
    "Eosinophils": 1,
    "Erythroblasts": 2,
    "IG": 3,
    "Lymphoblasts": 4,
    "Lymphocytes": 5,
    "Monocytes": 6,
    "Neutrophils": 7,
    "Platelet": 8   
    }
    num_classes = len(label_map)

    for client_number in range(num_clients):
        validation_dir = f'FL_noiid_split_40/validation/client_{client_number}/part{round_number}'
        for class_dir in os.listdir(validation_dir):
            class_dir_path = os.path.join(validation_dir, class_dir)
            if os.path.isdir(class_dir_path):
                for filename in os.listdir(class_dir_path):
                    img_path = os.path.join(class_dir_path, filename)
                    if os.path.isfile(img_path):
                        img = load_img(img_path, target_size=(224, 224))  
                        img = img_to_array(img) 
                        img = img/255.0

                        validation_images[client_number].append(img)

                        label = os.path.basename(os.path.dirname(img_path)) 
                        #label_encoded = to_categorical(label_map[label], num_classes=num_classes)
                        validation_labels[client_number].append(label)

    return validation_images, validation_labels




def move_images(source_dir, target_dir, percentage, client_number):
    if client_number in [1, 2, 4]:
        expected_subdirs = ['Basophils', 'Eosinophils', 'Erythroblasts', 'IG', 'Lymphoblasts', 'Lymphocytes', 'Monocytes', 'Neutrophils', 'Platelet']
    elif client_number == 3:
        expected_subdirs = ['Basophils', 'Eosinophils', 'Erythroblasts', 'IG', 'Lymphocytes', 'Monocytes', 'Neutrophils', 'Platelet']
    elif client_number == 0:
        expected_subdirs = ['Basophils', 'Eosinophils', 'Erythroblasts', 'IG', 'Lymphoblasts', 'Lymphocytes', 'Monocytes', 'Neutrophils']

    subdirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d)) and d in expected_subdirs]

    if set(expected_subdirs) != set(subdirs):
        print("Error: los subdirectorios en source_dir no coinciden con los subdirectorios esperados.")
        return

    for subdir in subdirs:
        subdir_source = os.path.join(source_dir, subdir)
        subdir_target = os.path.join(target_dir, subdir)

        os.makedirs(subdir_target, exist_ok=True)

        files = os.listdir(subdir_source)

        num_files_to_move = int(percentage * len(files))

        files_to_move = random.sample(files, num_files_to_move)

        for file_name in files_to_move:
            source_file = os.path.join(subdir_source, file_name)
            target_file = os.path.join(subdir_target, file_name)
            shutil.move(source_file, target_file)

        print(f'Number of images in {subdir_target} after moving: {len(os.listdir(subdir_target))}')