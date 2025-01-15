import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score


# Load the uploaded CSV files
accuracies_path = '../data/accuracies.csv'
losses_path = '../data/losses.csv'
predictions_path = '../data/predictions.csv'
true_data_path = '../data/spiral_data.csv'

# Read the files into pandas DataFrames
accuracies_df = pd.read_csv(accuracies_path)
losses_df = pd.read_csv(losses_path)
predictions_df = pd.read_csv(predictions_path)
true_data_df = pd.read_csv(true_data_path)

# Rename the column to "Loss" for clarity
losses_df.columns = ['Loss']

# Plot the loss values
plt.figure(figsize=(8, 5))
plt.plot(losses_df['Loss'], label='Loss', color='blue')
plt.title('Losses Over Time')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


