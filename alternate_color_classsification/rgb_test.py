import cv2
import torch
import torch.nn as nn
from torch.optim import Adam
import pandas as pd
import numpy as np
import torch.nn.functional as F
import datetime
import time
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, precision_score, recall_score

from multicell_classification.misc_utils import plot_accuracy, set_seed, plot_roc
from rgb_cnn_alt import RGBConvNetConcat, RGBConcatImageDataset

set_seed(42)
st = time.time()

df = pd.read_csv('groundtruths.csv')
label_mapping = {'rbc': 0, 'platelet': 1, 'wbc platelet': 2, 'wbc': 3}
df['label'] = df['label'].map(label_mapping)

# Split data
train_val_df, test_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
print(f'Training+Validation: {len(train_val_df)}, Test: {len(test_df)}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_weights = torch.tensor([1/136, 1/1104, 1/201, 1/206], dtype=torch.float).to(device)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
# Transforms
train_transform = transforms.Compose([
    transforms.Lambda(lambda img: transforms.ToTensor()(img) if not isinstance(img, torch.Tensor) else img),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])
test_transform = transforms.Compose([
    transforms.Lambda(lambda img: transforms.ToTensor()(img) if not isinstance(img, torch.Tensor) else img),
])
batch_size = 16
lr_factor = 0.9
roc_auc_val = 0.9
test_dataset = RGBConcatImageDataset(test_df, transforms=test_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


model_path = './saved_rgb_models/24-04_1329_92_model.pt'
model = torch.load(model_path, weights_only=False)
model.eval()
all_preds, all_labels, all_probs = [], [], []
total_loss = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        loss = loss_fn(outputs, labels)
        total_loss += loss.item() * labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())


test_acc = (np.array(all_preds) == np.array(all_labels)).mean() * 100
roc = roc_auc_score(all_labels, all_probs, multi_class='ovo')
f1 = f1_score(all_labels, all_preds, average='weighted')
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
loss = total_loss / len(test_dataset)
test_metrics = pd.DataFrame([{
    'Accuracy': round(test_acc, 4),
    'F1 Score': round(f1, 4),
    'ROC AUC': round(roc, 4),
    'Precision': round(precision, 4),
    'Recall': round(recall, 4),
    'Loss': round(loss, 4)
}])
# res_df = pd.DataFrame([results])
test_metrics.to_excel('alt_color_class_results.xlsx', index=False)

print("--- Test Set Evaluation ---")
print(f"Accuracy: {test_acc:.2f}%")
print(f"F1 Score: {f1:.4f}, ROC AUC: {roc:.4f}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")



print('Total time taken: ', time.time() - st)