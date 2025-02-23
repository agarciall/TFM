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
import copy
import gc 
from keras import backend as K 
import sys
import pandas as pd
from FL_utils_noiid import log_memory_and_time, MemoryUsageCallback, save_model_and_history, create_model, get_data_for_round, train_model, generate_validation_images, add_validation_images_for_round, move_images
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
import shutil




start_round = int(sys.argv[1])
end_round = int(sys.argv[2])
end_round = end_round + 1


# Create an experiment directory   
nom = "8_1_brightness_epochs_2_reload0_weighted"
Model = "ResNet50"
directory = os.path.join('FL_noiid_train', Model,  nom)
os.makedirs(directory, exist_ok=True)


validation_generators = [[] for _ in range(5)]

if start_round == 1:

    best_global_val_acc = 0
    print("Best global validation accuracy initialized to 0")
    global_history = {'global_val_acc': []}
    model, trainable_layer_indices = create_model()
    initial_weights = copy.deepcopy(model.get_weights())
    client_histories = {f'client_{i}_{j}': {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []} for i in range(5) for j in range(41)}
    memory_usage_df = pd.DataFrame(columns=['round', 'client', 'memory_usage'])
    weights_memory = []
    validation_images, validation_labels = generate_validation_images(start_round, 5)
    accuracies = {f'client_{i}': [] for i in range(5)}

else:
    global_history = {'global_val_acc': []}

    model, trainable_layer_indices = create_model()
    load_model = os.path.join(directory, f'model_round_{start_round-1}.h5')
    last_model = tf.keras.models.load_model(load_model)
    initial_weights = copy.deepcopy(last_model.get_weights())
    memory_usage_df = pd.DataFrame(columns=['round', 'client', 'memory_usage'])

    validation_images, validation_labels = generate_validation_images(start_round-1, 5)
    del last_model

    with open(os.path.join(directory, 'weights_memory.pkl'), 'rb') as f:
        weights_memory = pickle.load(f)
    with open(os.path.join(directory,'global_history.pkl'), 'rb') as f:
        global_history = pickle.load(f)
    with open(os.path.join(directory,'client_histories.pkl'), 'rb') as f:
        client_histories = pickle.load(f)
    with open(os.path.join(directory,'accuracies.pkl'), 'rb') as f:
        accuracies = pickle.load(f)
    with open(os.path.join(directory, 'best_global_val_acc.pkl'), 'rb') as f:
        best_global_val_acc = pickle.load(f)
    print(f"Best global validation accuracy loaded: {best_global_val_acc}")



total_clients = 5

dummy_image_path = "./dummy_image.jpeg"

label_map_path = "./label_map.txt.txt"

with open(label_map_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]




for round_number in range(start_round, end_round):
    print(f'Round {round_number} Training begins...')
    round_number = round_number
    round_weights = {}
    local_weights =[]
    for i in range(5):
        print(f'Client {i} Training begins...')
        client_number = i

        if client_number == 0:
            target_dir = f'/home/FL_noiid_split_40/train/client_{client_number}/part{round_number}/Platelet'
            if not os.listdir(target_dir):
                shutil.copy(dummy_image_path, target_dir)
        elif client_number == 3:
            target_dir = f'/home/FL_noiid_split_40/train/client_{client_number}/part{round_number}/Lymphoblasts'
            if not os.listdir(target_dir):
                shutil.copy(dummy_image_path, target_dir)

        train_data, validation_data, class_weights = get_data_for_round(client_number, round_number)
        client_model = model
        client_model.set_weights(initial_weights)





        # Train the model on the client's data
        history, weights, local_model = train_model(client_model, train_data, validation_data, trainable_layer_indices, round_number, client_number, memory_usage_df, class_weights, epochs=2)
        
        
        pes_parametres_client = sys.getsizeof(weights) / (1024 ** 2)
        round_weights[f'Client{client_number}_MB'] = pes_parametres_client


        weights = copy.deepcopy(local_model.get_weights())
        local_weights.append(weights)

        print(trainable_layer_indices)
        print(len(weights))


        # Save the client's training history
        client_histories[f'client_{client_number}_{round_number}']['train_loss'].extend(history.history['loss'])
        client_histories[f'client_{client_number}_{round_number}']['train_acc'].extend(history.history['accuracy'])
        client_histories[f'client_{client_number}_{round_number}']['val_loss'].extend(history.history['val_loss'])
        client_histories[f'client_{client_number}_{round_number}']['val_acc'].extend(history.history['val_accuracy'])



        #source_dir = f'/home/FL_noiid_split_40/train/client_{client_number}/part{round_number}'
        #target_dir = f'/home/FL_noiid_split_40/train/client_{client_number}/part{round_number+1}'

        #move_images(source_dir, target_dir, 0.6, client_number)

        
        
        del client_model
        K.clear_session()
        del train_data
        del validation_data
        gc.collect
    

    print(f'Round {round_number-1} Evaluation begins...')

    


    initial_weights = [np.mean([weight[w] for weight in local_weights], axis=0) for w in range(len(local_weights[0]))]


    print("Parameters aggregated")
    global_model = model
    global_model.set_weights(initial_weights)
    print("New model created")



    trainable_weights = [layer.get_weights() for layer in global_model.layers if layer.trainable]
    print(layer for layer in global_model.layers if layer.trainable)

    weights_bytes = pickle.dumps(trainable_weights)
    round_weights["Average_MB"] = sys.getsizeof(weights_bytes) / (1024 ** 2)

    weights_bytes = pickle.dumps(global_model.get_weights())
    round_weights["Whole_Model_MB"] = sys.getsizeof(weights_bytes) / (1024 ** 2)

    weights_memory.append(round_weights)

    gc.collect()

    print(f'Round {round_number} Aggregating ends...')


    if round_number not in [1, 11, 21, 31]:
        validation_images, validation_labels = add_validation_images_for_round(validation_images, validation_labels, round_number-1, 5) 
    # Log the memory usage and time

    for i in range(5):
        client_number = i
        test_images = np.array(validation_images[client_number])
        predictions = global_model.predict(test_images)
        predicted_classes = np.argmax(predictions, axis=1)
        predicted_class_names = [classes[c] for c in predicted_classes]
        print(predicted_class_names)

        test_labels = np.array(validation_labels[client_number])


        accuracy = np.mean(predicted_class_names == test_labels)

        print(f'Accuracy: {accuracy * 100:.2f}%')

        if f'client_{client_number}' in accuracies:
            accuracies[f'client_{client_number}'].append(accuracy)
        else:
            accuracies[f'client_{client_number}'] = [accuracy]





    last_accuracies = [accuracies[f'client_{i}'][-1] for i in range(5)]

    average_accuracy = sum(last_accuracies) / 5

    print(f'Average accuracy: {average_accuracy * 100:.2f}%')

    global_history['global_val_acc'].append(average_accuracy)

    if average_accuracy > best_global_val_acc:
        best_global_val_acc = average_accuracy
        global_model.save(os.path.join(directory, f'best_global_model_round_{round_number}.h5'))
        print(f'Round {round_number} Best global model saved...')

    print(f'Round {round_number} Aggregating begins...')
    

    memory = log_memory_and_time()

    
    # Save the log info to a pickle file
    with open(os.path.join(directory,'memory_history.pkl'), 'wb') as f:
        pickle.dump(memory, f)
    print(f'Round {round_number} Logging ends...')

    if round_number == end_round - 1:
        model.save(os.path.join(directory, f'model_round_{round_number}.h5'))
        memory_usage_df.to_csv(os.path.join(directory,f'memory_usage_df_{round_number}.csv'), index=False)
        with open(os.path.join(directory, 'best_global_val_acc.pkl'), 'wb') as f:
            pickle.dump(best_global_val_acc, f)
        with open(os.path.join(directory, f'client_histories.pkl'), 'wb') as f:
            pickle.dump(client_histories, f)
        with open(os.path.join(directory, f'accuracies.pkl'), 'wb') as f:
            pickle.dump(accuracies, f)
        with open(os.path.join(directory, f'global_history.pkl'), 'wb') as f:
            pickle.dump(global_history, f)
        with open(os.path.join(directory, 'weights_memory.pkl'), 'wb') as f:
            pickle.dump(weights_memory, f)
    
    tf.keras.backend.clear_session()
    K.clear_session()

#tf.config
#grow