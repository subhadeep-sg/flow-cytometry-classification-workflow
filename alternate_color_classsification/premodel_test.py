import os
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as tmodels
import torch.optim as optim
import torch.nn as nn
import time
import datetime
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, auc, precision_score, \
    recall_score
from multicell_classification.misc_utils import set_seed, get_current_time, plot_accuracy, mean_std_computation, \
    check_predictions
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from rgb_cnn_alt import RGBConvNetConcat, RGBConcatImageDataset

st = time.time()
set_seed(42)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Load and preprocess dataset
df = pd.read_csv('groundtruths.csv')
label_mapping = {'rbc': 0, 'platelet': 1, 'wbc platelet': 2, 'wbc': 3}
df['label'] = df['label'].map(label_mapping)

train_val_df, test_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
print(f'Training+Validation: {len(train_val_df)}, Test: {len(test_df)}')

train_transform = transforms.Compose([
    transforms.Lambda(lambda img: transforms.ToTensor()(img) if not isinstance(img, torch.Tensor) else img),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])
test_transform = transforms.Compose([
    transforms.Lambda(lambda img: transforms.ToTensor()(img) if not isinstance(img, torch.Tensor) else img),
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_weights = torch.tensor([1/136, 1/1104, 1/201, 1/206], dtype=torch.float).to(device)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

batch_size = 16
lr_factor = 0.9
roc_auc_val = 0.9

test_ds = RGBConcatImageDataset(test_df, transforms=test_transform)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# model_path = './saved_models/EfficientNet_25-04_0430_90_model.pt'
# model_path = './saved_models/MobileNetV2_25-04_0812_85_model.pt'
model_path = './saved_models/ResNet_25-04_0843_86_model.pt'


model = torch.load(model_path, weights_only=False)
model.eval()

all_preds, all_labels, all_probs, total_loss = [], [], [], 0.0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        loss = loss_fn(outputs, labels)
        total_loss += loss.item()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

test_acc = (np.array(all_preds) == np.array(all_labels)).mean() * 100
test_f1 = f1_score(all_labels, all_preds, average='weighted')
test_roc = roc_auc_score(all_labels, all_probs, multi_class='ovo')
test_precision = precision_score(all_labels, all_preds, average='weighted')
test_recall = recall_score(all_labels, all_preds, average='weighted')
test_loss = total_loss / len(test_loader)

print("\n--- Test Set Results ---")
print(f"Accuracy: {test_acc:.2f}%")
print(f"F1 Score: {test_f1:.4f}")
print(f"ROC AUC: {test_roc:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"Loss: {test_loss:.4f}")

results_df = pd.DataFrame([{
    'Accuracy': round(test_acc, 4),
    'F1': round(test_f1, 4),
    'ROC-AUC': round(test_roc, 4),
    'Precision': round(test_precision, 4),
    'Recall': round(test_recall, 4),
    'Loss': round(test_loss, 4)
}])
results_df.to_excel(f'ResNet_test_results.xlsx', index=False)

print('Total runtime:', time.time() - st)
