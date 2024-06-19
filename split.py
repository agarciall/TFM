import os
import shutil
import random

# Set the seed
random.seed(1234)

# Define the directories
original_dir = './MergeD'
train_dir = './train_subset'
test_dir = './test_subset'

# Define the classes
classes = ["Basophils", "Eosinophils", "Erythroblasts", "IG", "Lymphocytes", "Monocytes", "Neutrophils", "Platelet", "Lymphoblasts"]

# Create the train and test directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# For each class, split the data into train and test sets
for class_name in classes:
    # Create class-specific subdirectories in the train and test directories
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    # Get the list of all the images
    image_list = os.listdir(os.path.join(original_dir, class_name))
    # Shuffle the list
    random.shuffle(image_list)
    
    # Select only 10% of the data
    image_list = image_list[:int(len(image_list))]
    
    # Split the list into train and test
    split_point = int(len(image_list) * 0.8)
    train_list = image_list[:split_point]
    test_list = image_list[split_point:]
    
    # Copy the images into the appropriate directories
    for image in train_list:
        shutil.copy(os.path.join(original_dir, class_name, image), os.path.join(train_dir, class_name, image))
    for image in test_list:
        shutil.copy(os.path.join(original_dir, class_name, image), os.path.join(test_dir, class_name, image))