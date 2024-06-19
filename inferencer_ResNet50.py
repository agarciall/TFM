from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import glob
import os 

# Paths
test_dataset_path = "./test_subset"
path = "/home/FL_iid_train/ResNet50/8_1_brightness_epochs_2_reload0_weighted_DP"
model_path = os.path.join(path, "best_global_model_round_38.h5")
output_path = os.path.join(path, "predictions.csv")
label_map_path = "./label_map.txt.txt"  # Update this to your label map file path


# Load the model
model = load_model(model_path, compile = False)




with open(label_map_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Prepare the test data
test_images = []
test_files = []
for root, dirs, files in os.walk(test_dataset_path):
    for file in files:
        if file.endswith((".jpg", ".bmp")):  # Check for both .jpg and .bmp files
            img_path = os.path.join(root, file)
            try:
                image = load_img(img_path, target_size=(224, 224))
                image = img_to_array(image)
                image = image/255.0  # Normalizing the image
                test_images.append(image)
                test_files.append(img_path)
            except Exception as e:
                print(f"Skipping {img_path} due to error: {e}")
test_images = np.array(test_images)

# Make inference
predictions = model.predict(test_images)


# Prepare the results
results = []
for i, prediction in enumerate(predictions):
    top_2_indices = prediction.argsort()[-2:][::-1]
    top_2_values = prediction[top_2_indices]
    # Get the filename without the full path
    filename = os.path.basename(test_files[i])
    # Get the true class from the directory name
    true_class = os.path.basename(os.path.dirname(test_files[i]))
    results.append([filename, classes[top_2_indices[0]], top_2_values[0], classes[top_2_indices[1]], top_2_values[1], true_class])

# Save the results to a CSV file
df = pd.DataFrame(results, columns=["File_name", "Predicted Class", "Score1", "2nd Predicted Class", "Score2", "True Class"])
df.to_csv(output_path, index=False)