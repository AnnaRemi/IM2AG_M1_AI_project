import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score

# Load the uploaded CSV files
accuracies_path = 'accuracies.csv'
losses_path = 'losses.csv'
predictions_path = 'predictions.csv'
true_data_path = '../data/spiral_data.csv'

# Read the files into pandas DataFrames
accuracies_df = pd.read_csv(accuracies_path)
losses_df = pd.read_csv(losses_path)
predictions_df = pd.read_csv(predictions_path)
last_row_df = predictions_df.tail(1).squeeze().reset_index(drop=True)
true_data_df = pd.read_csv(true_data_path)

assert len(true_data_df) == len(last_row_df), "Lengths of true_data_df and last_row_df are not equal"

# Example true labels and predictions (binary classification)
true_data = true_data_df['y']  # True labels
predictions = last_row_df  # Predicted probabilities

# Calculate accuracy
accuracy = accuracy_score(true_data, [1 if p > 0.5 else 0 for p in predictions])
print(f"Accuracy: {accuracy * 100:.2f}%")



# Calculate class-wise accuracy
for class_label in np.unique(true_data):
    class_indices = np.where(true_data == class_label)[0]
    class_accuracy = accuracy_score(true_data[class_indices], predictions[class_indices])
    print(f"Accuracy for class {class_label}: {class_accuracy}")



# Calculate confusion matrix
cm = confusion_matrix(true_data, predictions)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=np.unique(true_data), yticklabels=np.unique(true_data))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

