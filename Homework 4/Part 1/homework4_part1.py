# -*- coding: utf-8 -*-
"""Homework4_Part1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1aIlzG32e31jAd-qHYHFOTGs0T59jBW-q
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_curve, auc

# Load the dataset
df = pd.read_csv('hw4_data.csv')


# Extract columns as numpy arrays
model_output = df['model_output'].to_numpy()
true_class = df['true_class'].to_numpy()
y_pred = df['prediction'].to_numpy()

# Compute confusion matrix
tn, fp, fn, tp = confusion_matrix(true_class, y_pred).ravel()

# Print results
print(f"True Positives: {tp}")
print(f"False Positives: {fp}")
print(f"True Negatives: {tn}")
print(f"False Negatives: {fn}")

precision = precision_score(true_class, y_pred)
recall = recall_score(true_class, y_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(true_class, model_output)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random chance line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# Find threshold for at least 90% recall
required_recall = 0.90
valid_indices = np.where(tpr >= required_recall)[0]

if len(valid_indices) > 0:
    min_fpr = np.min(fpr[valid_indices])
    print(f"Minimum False Positive Rate for at least 90% recall: {min_fpr:.4f}")
else:
    print("No threshold meets the 90% recall requirement.")