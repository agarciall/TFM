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
from FL_utils_iid import log_memory_and_time, MemoryUsageCallback, save_model_and_history, create_model, get_data_for_round, train_model, generate_validation_images, add_validation_images_for_round, move_images
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
from tensorflow.keras.utils import custom_object_scope

import tensorflow_privacy





start_round = int(sys.argv[1])
end_round = int(sys.argv[2])
end_round = end_round + 1


# Create an experiment directory   
nom = "8_1_brightness_epochs_2_reload60_weighted_miau"
Model = "ResNet50"
directory = os.path.join('FL_iid_train', Model,  nom)
os.makedirs(directory, exist_ok=True)


validation_generators = [[] for _ in range(5)]

if start_round == 1:

    best_global_val_acc = 0
    print("Best global validation accuracy initialized to 0")
    total_steps = 0
    global_epsilon = []
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
    class DPOptimizerClass(tf.keras.optimizers.Optimizer):
        def __init__(self, l2_norm_clip, noise_multiplier, num_microbatches, learning_rate):
            self.optimizer = DPKerasSGDOptimizer(
                l2_norm_clip=1.0,
                noise_multiplier=2.1,
                num_microbatches=1,
                learning_rate=0.001,
            )

        def get_optimizer(self):
            return self.optimizer

    
    model, trainable_layer_indices = create_model()
    load_model = os.path.join(directory, f'model_round_{start_round-1}.h5')
    last_model = tf.keras.models.load_model(load_model, compile = False)
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
    with open(os.path.join(directory, 'total_steps.pkl'), 'rb') as f:
        total_steps = pickle.load(f)
    with open(os.path.join(directory, 'global_epsilon.pkl'), 'rb') as f:
        global_epsilon = pickle.load(f)
    print(f"Best global validation accuracy loaded: {best_global_val_acc}")



total_clients = 5


label_map_path = "./label_map.txt.txt"

with open(label_map_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

noise_multiplier = 2.1
delta = 1e-5 


for round_number in range(start_round, end_round):
    print(f'Round {round_number} Training begins...')
    round_number = round_number
    round_weights = {}
    local_weights =[]
    for i in range(5):
        print(f'Client {i} Training begins...')
        client_number = i
        # Load the client's data for this round
        train_data, validation_data, class_weights = get_data_for_round(client_number, round_number)
        client_model = model
        client_model.set_weights(initial_weights)


        steps_per_round = len(train_data) * 2

        total_steps += steps_per_round

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



        source_dir = f'/home/FL_iid_split_40/train/client_{client_number}/part{round_number}'
        target_dir = f'/home/FL_iid_split_40/train/client_{client_number}/part{round_number+1}'

        move_images(source_dir, target_dir, 0.6)

        
        
        del client_model
        K.clear_session()
        del train_data
        del validation_data
        gc.collect
    


    epsilon, optimal_order = tensorflow_privacy.compute_dp_sgd_privacy(130*5*round_number, 8, noise_multiplier, round_number*5*2, delta)
    print(f'For delta = {delta} the current epsilon is: {epsilon}')
    global_epsilon.append(epsilon)

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
        with open(os.path.join(directory, 'global_epsilon.pkl'), 'wb') as f:
            pickle.dump(global_epsilon, f)
        with open(os.path.join(directory, 'total_steps.pkl'), 'wb') as f:
            pickle.dump(total_steps, f)
        with open(os.path.join(directory,'memory_history.pkl'), 'wb') as f:
            pickle.dump(memory, f)
    
    tf.keras.backend.clear_session()
    K.clear_session()

