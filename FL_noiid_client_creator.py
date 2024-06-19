import os
import shutil
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


classes = ["Basophils", "Eosinophils", "Erythroblasts", "IG", "Lymphoblasts", "Lymphocytes", "Monocytes", "Neutrophils", "Platelet"]
proportions = [(0.4, 0.0), (0.2, 0.2), (0.3, 0.1), (0.0, 0.4), (0.1, 0.3)]
client_class_counts = {i: {class_name: 0 for class_name in classes} for i in range(5)}


# Create client directories if they don't exist
for i in range(5):
    os.makedirs(f'./FL_noiid/client_{i}', exist_ok=True)

for class_name in classes:
    # Get HRD and PBC_DIB files
    hrd_files = [file for file in os.listdir(f'./train_subset/{class_name}') if file.startswith('HRD_')]
    pcb_dib_files = [file for file in os.listdir(f'./train_subset/{class_name}') if file.startswith('PBC_DIB_')]

    # Shuffle the files to ensure randomness
    random.shuffle(hrd_files)
    random.shuffle(pcb_dib_files)

    hrd_start = pcb_dib_start = 0
    for i, (hrd_prop, pcb_dib_prop) in enumerate(proportions):
        # Calculate the number of HRD and PBC_DIB files for this client
        os.makedirs(f'./FL_noiid/client_{i}/{class_name}', exist_ok=True)
        num_hrd_files = int(len(hrd_files) * hrd_prop)
        num_pcb_dib_files = int(len(pcb_dib_files) * pcb_dib_prop)

        # Copy HRD files
        for file in hrd_files[hrd_start:hrd_start + num_hrd_files]:
            shutil.copy(f'./train_subset/{class_name}/{file}', f'./FL_noiid/client_{i}/{class_name}/{file}')

        # Copy PBC_DIB files
        for file in pcb_dib_files[pcb_dib_start:pcb_dib_start + num_pcb_dib_files]:
            shutil.copy(f'./train_subset/{class_name}/{file}', f'./FL_noiid/client_{i}/{class_name}/{file}')

        # Update the start indices for the next client
        hrd_start += num_hrd_files
        pcb_dib_start += num_pcb_dib_files




# Generate the graph
client_indices = np.arange(5)
bar_width = 0.1

fig, ax = plt.subplots()

for i, class_name in enumerate(classes):
    class_counts = [client_class_counts[j][class_name] for j in range(5)]
    ax.bar(client_indices + i * bar_width, class_counts, bar_width, label=class_name)

ax.set_xlabel('Client')
ax.set_ylabel('Number of files')
ax.set_title('Number of files of each class for each client')
ax.set_xticks(client_indices + bar_width * (len(classes) - 1) / 2)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#ax.set_xticklabels([f'Client {i}' for i in range(5)])
ax.legend()

plt.savefig('client_class_counts.png')