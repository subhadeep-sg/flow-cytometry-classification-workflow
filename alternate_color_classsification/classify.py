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

# Transforms
train_transform = transforms.Compose([
    transforms.Lambda(lambda img: transforms.ToTensor()(img) if not isinstance(img, torch.Tensor) else img),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])
test_transform = transforms.Compose([
    transforms.Lambda(lambda img: transforms.ToTensor()(img) if not isinstance(img, torch.Tensor) else img),
])

# Cross-validation
k = 3
batch_size = 16
lr_factor = 0.9
roc_auc_val = 0.9
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class_weights = torch.tensor([1/136, 1/1104, 1/201, 1/206], dtype=torch.float).to(device)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

X_train_val = train_val_df.drop(columns=['label'])
y_train_val = train_val_df['label']
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
cv_results = []

def rgb_train(net, epochs, train_loader, valid_loader, optimizer, loss_fn, current_date, class_weights=None,
              save=False, set_highest_acc=70.0, set_highest_f1=0.7, set_roc_auc=0.7, device=torch.device('cuda'),
              verbose=False):
    print('--------Training start-------------')
    train_st = time.time()
    final_res = {'f1': set_highest_f1,
                 'accuracy': set_highest_acc,
                 'roc-auc': set_roc_auc,
                 'precision': None,
                 'recall': None,
                 'loss': None,
                 'model path': None,
                 'class weights': class_weights,
                 'all_pred': None,
                 'all_labels': None,
                 'all_probs': None,
                 }
    highest_f1 = set_highest_f1
    highest_acc = set_highest_acc
    highest_roc_auc = set_roc_auc
    model_history = {'train': [], 'valid': []}
    for epoch in range(epochs):
        running_loss, val_running_loss = 0, 0
        total_samples, val_total = 0, 0
        pred_correct, val_pred_correct = 0, 0
        all_pred, all_labels = [], []
        val_pred, val_labels = [], []
        all_probs, val_probs = [], []

        net.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            probs = F.softmax(outputs, dim=1)
            _, pred = torch.max(outputs, 1)

            pred_correct += (pred == labels).sum().item()
            total_samples += labels.size(0)

            all_pred.append(pred.cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        all_pred = np.concatenate(all_pred, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        net.eval()
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                probs = F.softmax(outputs, dim=1)
                _, pred = torch.max(outputs, 1)

                val_pred_correct += (pred == labels).sum().item()
                val_total += labels.size(0)

                val_pred.append(pred.cpu().numpy())
                val_probs.append(probs.cpu().numpy())
                val_labels.append(labels.cpu().numpy())

                validation_loss = loss_fn(outputs, labels)
                val_running_loss += validation_loss.item()

        val_pred = np.concatenate(val_pred, axis=0)
        val_probs = np.concatenate(val_probs, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)

        accuracy = (pred_correct / total_samples) * 100
        validation_accuracy = (val_pred_correct / val_total) * 100
        roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovo')
        val_f1 = f1_score(val_labels, val_pred, average='weighted')
        val_roc_auc = roc_auc_score(val_labels, val_probs, multi_class='ovo')
        val_precision = precision_score(val_labels, val_pred, average='weighted')
        val_recall = recall_score(val_labels, val_pred, average='weighted')

        model_history['train'].append(accuracy)
        model_history['valid'].append(validation_accuracy)

        if val_roc_auc > highest_roc_auc:
            highest_acc = validation_accuracy
            highest_roc_auc = val_roc_auc
            highest_f1 = val_f1
            highest_precision = val_precision
            highest_recall = val_recall
            final_res.update({
                'f1': round(val_f1, 4),
                'precision': round(val_precision, 4),
                'recall': round(val_recall, 4),
                'accuracy': round(validation_accuracy, 4),
                'loss': round(val_running_loss, 5),
                'all_pred': val_pred,
                'all_labels': val_labels,
                'all_probs': val_probs
            })
            if save:
                os.makedirs('saved_rgb_models', exist_ok=True)
                save_path = f'./saved_rgb_models/{current_date}_{round(val_roc_auc * 100)}_model.pt'
                torch.save(net, save_path)
                final_res['model path'] = save_path

        if verbose:
            print(f'Epoch:{epoch}, running loss:{running_loss:.4f}, train accuracy:{accuracy:.4f}, '
                  f'validation loss:{val_running_loss:.4f}, valid accuracy:{validation_accuracy:.4f}')
            print(f"Training: ROC-AUC: {roc_auc:.4f}")
            print(f"Validation: ROC-AUC: {val_roc_auc:.4f}, F1: {val_f1:.4f}")

    print(f'---------End training, time taken:{time.time() - train_st}-------------')

    if highest_roc_auc > set_roc_auc and verbose:
        print(f'Highest validation accuracy: ', highest_acc)
        print(f'Highest ROC-AUC score: ', highest_roc_auc)
        print(f'Highest F1-score: ', highest_f1)
        if save:
            print(f'Model saved at {final_res["model path"]}')

    return net, final_res, model_history

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_val, y_train_val)):
    print(f'Fold {fold+1}/{k}')
    train_df = train_val_df.iloc[train_idx].copy()
    val_df = train_val_df.iloc[val_idx].copy()

    train_dataset = RGBConcatImageDataset(train_df, transforms=train_transform)
    val_dataset = RGBConcatImageDataset(val_df, transforms=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = RGBConvNetConcat(in_channels=3, num_classes=4, img_width=896, img_height=224).to(device)
    optimizer = Adam(model.parameters(), lr=0.001 * lr_factor, weight_decay=1e-4)

    today = datetime.datetime.now().strftime("%d-%m_%H%M")
    model, results, history = rgb_train(
        net=model, epochs=50,
        optimizer=optimizer, loss_fn=loss_fn,
        train_loader=train_loader, valid_loader=val_loader,
        device=device, set_roc_auc=roc_auc_val,
        current_date=today, class_weights=class_weights,
        save=True, verbose=True
    )

    plot_accuracy(history, title=f"Fold {fold+1}")
    plot_roc(results, class_names=list(label_mapping.keys()))
    cv_results.append(results)

cv_df = pd.DataFrame.from_records(cv_results)
cv_df.to_csv('cross_validation_rgb_results.csv', index=False)
print("Saved CV results to cross_validation_rgb_results.csv")

print('Total time taken: ', time.time() - st)
