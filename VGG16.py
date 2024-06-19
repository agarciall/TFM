from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import pickle
import psutil
import time
import threading
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from datetime import datetime
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras
from sklearn.utils import class_weight
import numpy as np
#import datetime


seed = 123
current_time =  datetime.now().strftime('%Y%m%d_%H%M%S')
directory = os.path.join('VGG16', current_time)
checkpoint_path = os.path.join(directory, 'VGG16_best_{epoch:02d}.h5')
os.makedirs(directory, exist_ok=True)


class MemoryCallback(Callback):
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        self.thread = threading.Thread(target=self.log_memory_usage)
        self.thread.start()
        self.training_ended = False

    def log_memory_usage(self):
        with open('memory_usage_VGG16.txt', 'a') as f:
            while not self.training_ended:
                elapsed_time = time.time() - self.start_time
                memory = psutil.virtual_memory()
                memory_usage_percent = memory.percent
                memory_usage_kb = memory.used / 1024
                f.write(f"Elapsed time: {datetime.timedelta(seconds=elapsed_time)}, Memory usage: {memory_usage_percent}% ({memory_usage_kb} KB)\n")
                time.sleep(180)  # Log every

    def on_train_end(self, logs=None):
        self.training_ended = True
        self.thread.join()

    def log_memory_and_time():
        process = psutil.Process(os.getpid())
        print(f"Memory usage: {process.memory_info().rss / 1024 ** 2} MB")
        print(f"Time: {time.time()}")

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:  # Save model every 10 epochs
            self.model.save(f'model_at_epoch_{epoch}.h5')

class HistoryCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        epoch = epoch
        history_path = os.path.join(directory, f'history_VGG16_epoch_{epoch}.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(self.model.history.history, f)


history_callback = HistoryCallback()

checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')


# Define the parameters for the ImageDataGenerator
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

# The directory where all the classes are located
data_dir = './train_subset'

# Load the training data


train_generator = datagen.flow_from_directory(
    data_dir, 
    target_size   = (224, 224), 
    batch_size    = 8,
    interpolation = "bilinear",
    subset        = "training",
    seed          = seed, 
    shuffle       = True)


# Load the validation data
validation_data = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),  
    interpolation = "bilinear",
    batch_size= 1,
    subset='validation',  
    seed = seed,
    shuffle = False)



class_weights = class_weight.compute_class_weight('balanced', np.unique(train_generator.classes), train_generator.classes)
class_weights = dict(enumerate(class_weights))


baseModel = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Construct the head of the model
headModel = baseModel.output
headModel = layers.AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = layers.Flatten(name="flatten")(headModel)
headModel = layers.Dense(64, activation="relu")(headModel)
headModel = layers.Dropout(0.5)(headModel)
headModel = layers.Dense(9, activation="softmax")(headModel)  # Adjusted to 8 classes

model = models.Model(inputs=baseModel.input, outputs=headModel)

total_layers = len(model.layers)

trainable_layers = int(0.3 * total_layers)

# Set the last 30% of layers as trainable and get their indices
for i in range(-trainable_layers, 0):
    model.layers[i].trainable = True


# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=validation_data,
    validation_steps=len(validation_data),
    epochs=60,
    class_weight = class_weights,
    callbacks=[MemoryCallback(), checkpoint, history_callback]  # Add the checkpoint and history callbacks
)

print("Training finished.")

# 1. Save the final model
model_path = os.path.join(directory, 'final_model_VGG1650.h5')
model.save(model_path)

print("Model saved.")

# 2. Save the history
history_path = os.path.join(directory, 'history_VGG1650.pkl')
with open(history_path, 'wb') as f:
    pickle.dump(history.history, f)

print("History saved.")




import numpy as np
from keras.preprocessing.image import save_img

# Define the number of images to save per class and per dataset
num_images = 5

# Define the datasets
datasets = ['HRD', 'PBC_DIB']

# Define the classes
classes = ["Basophils", "Eosinophils", "Erythroblasts", "IG", "Lymphocytes", "Monocytes", "Neutrophils", "Platelet", "Lymphoblasts"]

# Create a dictionary to keep track of how many images have been saved per class and per dataset
saved_images = {class_name: {dataset: 0 for dataset in datasets} for class_name in classes}

# Iterate over the train generator
for images, labels in train_generator:
    # Iterate over each image and its corresponding label
    for image, label in zip(images, labels):
        # Get the class name
        class_name = classes[np.argmax(label)]
        
        # Get the dataset name
        dataset_name = 'HRD' if 'HRD' in train_generator.filenames[train_generator.batch_index] else 'PBC_DIB'
        
        # If enough images have been saved for this class and dataset, continue to the next image
        if saved_images[class_name][dataset_name] >= num_images:
            continue
        
        # Save the image
        image_path = os.path.join(directory, f'{class_name}_{dataset_name}_{saved_images[class_name][dataset_name]}.png')
        save_img(image_path, image)
        
        # Increment the count of saved images for this class and dataset
        saved_images[class_name][dataset_name] += 1
    
    # If enough images have been saved for all classes and datasets, break the loop
    if all(all(count >= num_images for count in class_dict.values()) for class_dict in saved_images.values()):
        break