import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np
import tensorflow as tf
import string
import matplotlib.pyplot as plt

# Define constants
RANDOM_SEED = 42
dataset = 'output/hasil_deteksi_tangan.csv'

# Load the model
model = tf.keras.models.load_model('output/mymodel.h5')

# Load your test data
X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))

# Display the total number of samples for each class
# Define the class labels from A to Z, excluding J and Z
class_labels = [char for char in string.ascii_uppercase if char not in ['J', 'Z']]

# Ensure the number of classes matches the labels
assert len(class_labels) == len(np.unique(y_dataset)), "Number of classes does not match the number of labels."

# Map the numeric classes to the alphabetic labels
y_dataset_labels = [class_labels[i] for i in y_dataset]

unique, counts = np.unique(y_dataset_labels, return_counts=True)
class_counts = dict(zip(unique, counts))

print("Total number of samples for each class:")
for cls, count in class_counts.items():
    print(f"Class {cls}: {count} samples")

# Plot the total number of samples for each class
plt.figure(figsize=(10, 7))
sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()), palette='viridis')
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.title('Total Number of Samples for Each Class')
plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.80, random_state=RANDOM_SEED)

# Make predictions
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Visualize classification report as a heatmap
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

plt.figure(figsize=(10, 7))
sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='Blues')
plt.title('Classification Report Heatmap')
plt.show()

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
