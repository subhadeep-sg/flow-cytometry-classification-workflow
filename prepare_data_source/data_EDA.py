import os
import matplotlib.pyplot as plt
import pandas as pd


color_df = pd.read_csv('../color_classification/groundtruths.csv')
label_column = color_df['label'].tolist()
print(color_df['label'].value_counts())

# Define dataset root
dataset_dir = '../multicell_classification/classify_dataset'
splits = ['train', 'test']
categories = ['cluster_revised', 'non_cluster_revised']

# Store counts
summary = {}

for split in splits:
    split_path = os.path.join(dataset_dir, split)
    summary[split] = {}
    for category in categories:
        category_path = os.path.join(split_path, category)
        if os.path.exists(category_path):
            count = len([f for f in os.listdir(category_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        else:
            count = 0
        summary[split][category] = count

# Print results
print("Dataset Summary:\n")
for split in summary:
    for category in summary[split]:
        print(f"{split}/{category}: {summary[split][category]} images")

# Plotting
labels = []
counts = []
for split in summary:
    for category in summary[split]:
        labels.append(f"{split}-{category}")
        counts.append(summary[split][category])
