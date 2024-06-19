import pickle
import os
import numpy as np
import matplotlib.pyplot as plt



directory = r"C:\Users\agarciall\Desktop\TFM\Federated\iid\8_1_brightness_epochs_2_reload0_weighted_DP"
directory_path = f"{directory}/figures"

# Check if the directory exists, if not, create it
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
with open(os.path.join(directory, 'accuracies.pkl'), 'rb') as f:
    accuracies = pickle.load(f)

print(accuracies)

with open(os.path.join(directory, 'global_history.pkl'), 'rb') as f:
    global_history = pickle.load(f)
print(global_history)

with open(os.path.join(directory, 'weights_memory.pkl'), 'rb') as f:
    weights_memory = pickle.load(f)
print(weights_memory)

with open(os.path.join(directory, 'client_histories.pkl'), 'rb') as f:
    client_histories = pickle.load(f)
print(client_histories)


### GRAFIC CLIENT_HISTORIES ###

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

median_patch = mpatches.Patch(color='blue', alpha=0.1, label='Area Under Clients Median')

# Create a figure with 4 subplots
fig, axs = plt.subplots(4, figsize=(10, 20))

# Define client labels
client_labels = ["Client 1", "Client 2", "Client 3", "Client 4", "Client 5"]

# Prepare lists to store metric values for all clients and rounds
all_val_loss = []
all_val_acc = []
all_train_loss = []
all_train_acc = []

for client_num in range(5):
    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []

    for round_num in range(1, 41):
        client_round = f"client_{client_num}_{round_num}"
        if client_round in client_histories:
            history = client_histories[client_round]

            # Append the second value of each metric for the current round
            if history['val_loss']:
                val_loss.append(history['val_loss'][1])
            else:
                val_loss.append(np.nan)
            if history['val_acc']:
                val_acc.append(history['val_acc'][1])
            else:
                val_acc.append(np.nan)
            if history['train_loss']:
                train_loss.append(history['train_loss'][1])
            else:
                train_loss.append(np.nan)
            if history['train_acc']:
                train_acc.append(history['train_acc'][1])
            else:
                train_acc.append(np.nan)

    # Plot the metrics for the current client
    axs[0].plot(range(1, 41), val_loss, label=client_labels[client_num])
    axs[1].plot(range(1, 41), val_acc, label=client_labels[client_num])
    axs[2].plot(range(1, 41), train_loss, label=client_labels[client_num])
    axs[3].plot(range(1, 41), train_acc, label=client_labels[client_num])
    
    for ax in axs:
        ax.set_xlim(1, 40)

    # Append the metric values for all clients and rounds
    all_val_loss.append(val_loss)
    all_val_acc.append(val_acc)
    all_train_loss.append(train_loss)
    all_train_acc.append(train_acc)

median_val_loss = np.nanmedian(all_val_loss, axis=0)
global_val_acc = global_history['global_val_acc']
global_val_acc_array = np.array(global_val_acc)
median_val_acc = np.nanmedian(all_val_acc, axis=0)
median_train_loss = np.nanmedian(all_train_loss, axis=0)
median_train_acc = np.nanmedian(all_train_acc, axis=0)

axs[0].fill_between(range(1, 41), median_val_loss, color='blue', alpha=0.1)
axs[1].fill_between(range(1, 41), median_val_acc, color='blue', alpha=0.1)

# Plot the global validation accuracy and fill the area between it and the median line
axs[1].plot(range(1, 41), global_val_acc, label='Global Validation Accuracy', color='violet')
axs[1].fill_between(range(1, 41), median_val_acc, global_val_acc, where=(global_val_acc > median_val_acc), color='violet', alpha=0.1, interpolate=True)

axs[2].fill_between(range(1, 41), median_train_loss, color='blue', alpha=0.1)
axs[3].fill_between(range(1, 41), median_train_acc, color='blue', alpha=0.1)



# Set titles and legends for the subplots
axs[0].set_title('Validation Loss')
axs[0].legend()
axs[1].set_title('Validation Accuracy')
axs[1].legend()
axs[2].set_title('Train Loss')
axs[2].legend()
axs[3].set_title('Train Accuracy')
axs[3].legend()


for ax in axs:
    # Get the existing legend entries
    handles, labels = ax.get_legend_handles_labels()
    
    # Add the median_patch to the legend entries
    handles.append(median_patch)
    labels.append('Area Under Clients Median')
    
    # Set the legend with the updated entries
    ax.legend(handles, labels)


import matplotlib.patches as mpatches

# Create a Patch object for the median area
median_patch = mpatches.Patch(color='blue', alpha=0.1, label='Area Under Clients Median')

# Create a figure and plot for each metric
metrics = ['Validation Loss', 'Validation Accuracy', 'Train Loss', 'Train Accuracy']
all_metrics = [all_val_loss, all_val_acc, all_train_loss, all_train_acc]

for i, metric in enumerate(metrics):
    fig, ax = plt.subplots(figsize=(10, 5))
    for client_num in range(5):
        ax.plot(range(1, len(all_metrics[i][client_num]) + 1), all_metrics[i][client_num], label=client_labels[client_num])
    median_metric = np.nanmedian(all_metrics[i], axis=0)
    ax.fill_between(range(1, len(median_metric) + 1), median_metric, color='blue', alpha=0.1)
    
    # Add the Global Validation line to the Validation Accuracy plot
    if metric == 'Validation Accuracy':
        ax.plot(range(1, len(global_val_acc) + 1), global_val_acc, label='Global Validation', color = "violet")
        ax.fill_between(range(1, len(global_val_acc) + 1), global_val_acc, median_metric, color='violet', alpha=0.1)
    
    ax.set_title(f"{metric}")
    
    handles, labels = ax.get_legend_handles_labels()
    
    # Add the median_patch to the legend entries
    handles.append(median_patch)
    labels.append('Area Under Clients Median')
    
    ax.legend(handles, labels, loc='upper left')
    
    ax.set_xlim(left=1)
    
    plt.tight_layout()
    plt.savefig(f"{directory_path}/{metric.replace(' ', '_')}.png")
    plt.close(fig)  



