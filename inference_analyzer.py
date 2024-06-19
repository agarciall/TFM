

import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import os

# Load the inference CSV file
dir = "/home/FL_iid_train/ResNet50/8_1_brightness_epochs_2_reload0_weighted_DP"
df = pd.read_csv(os.path.join(dir, 'predictions.csv'))

# Assuming class_names is a list of your class names
class_names = ["Basophils",
"Eosinophils",
"Erythroblasts",
"IG",
"Lymphoblasts",
"Lymphocytes",
"Monocytes",
"Neutrophils",
"Platelet"]

# Create a dictionary that maps class numbers to names
class_map = {i: name for i, name in enumerate(class_names)}

# Now you can use this dictionary to get the class name from a class number
class_number = 0  # Replace this with your actual class number
class_name = class_map[class_number]


# Calculate the confusion matrix
y_true = df['True Class']
y_pred = df['Predicted Class']

# Calculate the confusion matrix
cm = confusion_matrix(y_true, y_pred, labels =class_names)

# Plot the confusion matrix
plt.figure(figsize=(15, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=45)

# Save the plot as an image file
plt.savefig(os.path.join(dir, 'confusion_matrix.png'))



total_TP = total_FP = total_TN = total_FN = 0

# For each class
for i in range(len(class_names)):
    # Consider the current class as the positive class
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP
    FN = cm[i, :].sum() - TP
    TN = cm.sum() - (FP + FN + TP)
    
    # Add the counts to the total counts
    total_TP += TP
    total_FP += FP
    total_TN += TN
    total_FN += FN





# Calculate accuracy, F1 score, and recall
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')  # Use weighted average if you have imbalanced classes
recall = recall_score(y_true, y_pred, average='weighted')  # Use weighted average if you have imbalanced classes
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=1)
# Save the results to a .txt file
# Save the results to a .txt file
with open(os.path.join(dir, 'results.txt'), 'w') as f:
    f.write(f"TOTAL VALUES OF VALIDATION:\n")
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Total TP: {total_TP}\n")
    f.write(f"Total FP: {total_FP}\n")
    f.write(f"Total TN: {total_TN}\n")
    f.write(f"Total FN: {total_FN}\n\n\n")
    for class_name, metrics in report.items():
        if isinstance(metrics, dict):
            f.write(f"Class: {class_name}\n")
            f.write(f"Precision: {metrics['precision']}\n")
            f.write(f"Recall: {metrics['recall']}\n")
            f.write(f"F1 Score: {metrics['f1-score']}\n")
            f.write(f"Support: {metrics['support']}\n\n\n")
        else:
            f.write(f"{class_name}: {metrics}\n\n\n")