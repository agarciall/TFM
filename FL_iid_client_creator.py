import os
import shutil
import random
import matplotlib.pyplot as plt
import numpy as np

# Define the classes
classes = [d for d in os.listdir('./train_subset') if os.path.isdir(os.path.join('./train_subset', d))]

# Number of clients
num_clients = 5

# Create a directory for each client
for i in range(num_clients):
    os.makedirs(f'./FL_iid_nou/client_{i}', exist_ok=True)

random.seed(1234)

# Initialize a dictionary to store the count of each class for each client
client_class_counts = {i: {class_name: {'HRD': 0, 'PBC_DIB': 0} for class_name in classes} for i in range(num_clients)}

# Distribute the images to each client
for class_name in classes:
    hrd_files = [file for file in os.listdir(f'./train_subset/{class_name}') 
               if file.endswith(('.png', '.bmp', '.jpg')) and file.startswith('HRD_')]
    pcb_dib_files = [file for file in os.listdir(f'./train_subset/{class_name}') 
               if file.endswith(('.png', '.bmp', '.jpg')) and file.startswith('PBC_DIB_')]
    
    # Shuffle the lists
    random.shuffle(hrd_files)
    random.shuffle(pcb_dib_files)

    # Distribute files among clients
    for i in range(num_clients):
        hrd_files_for_client = hrd_files[i::num_clients]
        pcb_dib_files_for_client = pcb_dib_files[i::num_clients]
        
        os.makedirs(f'./FL_iid/client_{i}/{class_name}', exist_ok=True)
        
        # Copy files to client directory
        for file in hrd_files_for_client:
            shutil.copy(f'./train_subset/{class_name}/{file}', f'./FL_iid/client_{i}/{class_name}/{file}')
        for file in pcb_dib_files_for_client:
            shutil.copy(f'./train_subset/{class_name}/{file}', f'./FL_iid/client_{i}/{class_name}/{file}')
        
        # Update the count of the class for the client
        client_class_counts[i][class_name]['HRD'] += len(hrd_files_for_client)
        client_class_counts[i][class_name]['PBC_DIB'] += len(pcb_dib_files_for_client)


import os
import shutil
import random
import matplotlib.pyplot as plt
import numpy as np

# Define the classes
classes = [d for d in os.listdir('./train_subset') if os.path.isdir(os.path.join('./train_subset', d))]

# Number of clients
num_clients = 5

# Create a directory for each client
for i in range(num_clients):
    os.makedirs(f'./FL_iid_nou/client_{i}', exist_ok=True)

random.seed(1234)

# Initialize a dictionary to store the count of each class for each client
client_class_counts = {i: {class_name: {'HRD': 0, 'PBC_DIB': 0} for class_name in classes} for i in range(num_clients)}

# Distribute the images to each client
for class_name in classes:
    hrd_files = [file for file in os.listdir(f'./train_subset/{class_name}') 
               if file.endswith(('.png', '.bmp', '.jpg')) and file.startswith('HRD_')]
    pcb_dib_files = [file for file in os.listdir(f'./train_subset/{class_name}') 
               if file.endswith(('.png', '.bmp', '.jpg')) and file.startswith('PBC_DIB_')]
    
    # Shuffle the lists
    random.shuffle(hrd_files)
    random.shuffle(pcb_dib_files)

    # Distribute files among clients
    for i in range(num_clients):
        hrd_files_for_client = hrd_files[i::num_clients]
        pcb_dib_files_for_client = pcb_dib_files[i::num_clients]
        
        os.makedirs(f'./FL_iid/client_{i}/{class_name}', exist_ok=True)
        
        # Copy files to client directory
        for file in hrd_files_for_client:
            shutil.copy(f'./train_subset/{class_name}/{file}', f'./FL_iid/client_{i}/{class_name}/{file}')
        for file in pcb_dib_files_for_client:
            shutil.copy(f'./train_subset/{class_name}/{file}', f'./FL_iid/client_{i}/{class_name}/{file}')
        
        # Update the count of the class for the client
        client_class_counts[i][class_name]['HRD'] += len(hrd_files_for_client)
        client_class_counts[i][class_name]['PBC_DIB'] += len(pcb_dib_files_for_client)


import pickle

# Save the client_class_counts dictionary into a pickle file
with open('client_class_counts.pkl', 'wb') as f:
    pickle.dump(client_class_counts, f)

# Generate the abundance plots
for i in range(num_clients):
    labels = classes
    hrd_counts = [client_class_counts[i][class_name]['HRD'] for class_name in classes]
    pbc_dib_counts = [client_class_counts[i][class_name]['PBC_DIB'] for class_name in classes]

    x = np.arange(len(labels))  # the label locations

    fig, ax = plt.subplots()
    rects1 = ax.bar(x, hrd_counts, label='HRD')
    rects2 = ax.bar(x, pbc_dib_counts, bottom=hrd_counts, label='PBC_DIB')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Abundance')
    ax.set_title(f'Class Abundance for Client {i}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)
    ax.legend()

    fig.tight_layout()

    plt.savefig(f'./FL_iid/client_{i}/abundance.png')
    plt.close()