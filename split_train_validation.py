import os
import shutil
import numpy as np

data_dir = '/home/FL_noiid_split_40'
train_dir = '/home/FL_noiid_split_40/train'
valid_dir = '/home/FL_noiid_split_40/validation'

np.random.seed(66)

# Get the list of all client subdirectories
client_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

# Loop over each client directory
for client_dir in client_dirs:
    # Get the list of all part subdirectories in the client directory
    part_dirs = [d for d in os.listdir(os.path.join(data_dir, client_dir)) if os.path.isdir(os.path.join(data_dir, client_dir, d))]

    # Loop over each part directory
    for part_dir in part_dirs:
        # Get the list of all class subdirectories in the part directory
        class_dirs = [d for d in os.listdir(os.path.join(data_dir, client_dir, part_dir)) if os.path.isdir(os.path.join(data_dir, client_dir, part_dir, d))]

        # Loop over each class directory
        for class_dir in class_dirs:
            # Get the list of all image files in the class directory
            files = os.listdir(os.path.join(data_dir, client_dir, part_dir, class_dir))

            # Shuffle the list of files
            np.random.shuffle(files)

            # Split the files into a training set and a validation set
            train_files = files[:int(len(files)*0.8)]
            valid_files = files[int(len(files)*0.8):]

            # Create the corresponding client/part/class directory in the train and valid directories
            os.makedirs(os.path.join(train_dir, client_dir, part_dir, class_dir), exist_ok=True)
            os.makedirs(os.path.join(valid_dir, client_dir, part_dir, class_dir), exist_ok=True)

            # Move the training files to the train directory
            for file in train_files:
                shutil.move(os.path.join(data_dir, client_dir, part_dir, class_dir, file), os.path.join(train_dir, client_dir, part_dir, class_dir, file))

            # Move the validation files to the valid directory
            for file in valid_files:
                shutil.move(os.path.join(data_dir, client_dir, part_dir, class_dir, file), os.path.join(valid_dir, client_dir, part_dir, class_dir, file))