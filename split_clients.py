import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(1234)

base_dir = f"./FL_noiid"
new_base_dir = f"./FL_noiid_split_40"

# Get the list of clients
clients = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# For each client
for client in clients:
    client_dir = os.path.join(base_dir, client)

    # Get the list of classes
    classes = [d for d in os.listdir(client_dir) if os.path.isdir(os.path.join(client_dir, d))]

    # For each class
    for class_name in classes:
        class_dir = os.path.join(client_dir, class_name)

        # Get the list of images
        images = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]

        # Split the images into 40 parts
        split_images = np.array_split(images, 40)

        # For each part
        for i, part_images in enumerate(split_images):
            # Create a new directory for this part in the new base directory
            part_dir = os.path.join(new_base_dir, client, f'part{i+1}', class_name)
            os.makedirs(part_dir, exist_ok=True)

            # Copy the images to the new directory
            for image in part_images:
                shutil.copy(os.path.join(class_dir, image), os.path.join(part_dir, image))