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
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, MobileNet_V2_Weights, mobilenet_v2, \
    ResNet50_Weights, resnet50
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



def train(cnn, epochs, class_weights, train_loader, valid_loader, optimizer, lr_scheduler, loss_fn,
          save=False, current_date=None, set_highest_acc=70.0, set_highest_f1=0.6,
          device=torch.device('cuda'), verbose=False):
    if verbose:
        print('Training start....')
    train_start = time.time()
    highest_acc = set_highest_acc
    highest_f1 = set_highest_f1
    model_history = {'train': [], 'valid': []}
    running_record = {'f1': set_highest_f1, 'accuracy': set_highest_acc, 'roc-auc': None,
                      'recall': None, 'precision': None, 'loss': None,
                      'class_weights': class_weights, 'optimizer': optimizer, 'model path': None}

    for epoch in range(epochs):
        cnn.train()
        running_loss, val_running_loss = 0.0, 0.0
        total_samples, val_total = 0, 0
        pred_correct, val_pred_correct = 0, 0

        all_pred, all_labels, all_probs = [], [], []
        val_pred, val_labels, val_probs = [], [], []

        for inputs, labels in train_loader:
            inputs, labels = inputs.float().to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = cnn(inputs)
            probs = F.softmax(outputs, dim=1)
            _, pred = torch.max(outputs, 1)

            pred_correct += (pred == labels).sum().item()
            total_samples += labels.size(0)

            all_pred.append(pred.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            if lr_scheduler:
                lr_scheduler.step()

            running_loss += loss.item()

        all_pred = np.concatenate(all_pred, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        cnn.eval()
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = cnn(inputs)
                prob = F.softmax(outputs, dim=1)
                _, pred = torch.max(outputs, 1)

                val_pred_correct += (pred == labels).sum().item()
                val_total += labels.size(0)

                val_pred.append(pred.cpu().numpy())
                val_labels.append(labels.cpu().numpy())
                val_probs.append(prob.detach().cpu().numpy())

                val_running_loss += loss_fn(outputs, labels).item()

        val_pred = np.concatenate(val_pred, axis=0)
        val_probs = np.concatenate(val_probs, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)

        accuracy = (pred_correct / total_samples) * 100
        validation_accuracy = (val_pred_correct / val_total) * 100

        val_f1 = f1_score(val_labels, val_pred, average='macro')
        val_roc_auc = roc_auc_score(val_labels, val_probs, multi_class='ovo')
        val_precision = precision_score(val_labels, val_pred, average='macro')
        val_recall = recall_score(val_labels, val_pred, average='macro')

        model_history['train'].append(accuracy)
        model_history['valid'].append(validation_accuracy)

        if val_f1 > highest_f1:
            highest_acc = validation_accuracy
            highest_f1 = val_f1
            running_record['f1'] = round(val_f1, 4)
            running_record['accuracy'] = round(validation_accuracy, 4)
            running_record['precision'] = round(val_precision, 4)
            running_record['recall'] = round(val_recall, 4)
            running_record['roc-auc'] = round(val_roc_auc, 4)
            running_record['loss'] = round(val_running_loss, 4)
            if save:
                os.makedirs('saved_models', exist_ok=True)
                model_path = f'./saved_models/{cnn.__class__.__name__}_{current_date}_{round(highest_acc)}_model.pt'
                torch.save(cnn, model_path)
                running_record['model path'] = model_path

        if verbose:
            print(
                f"Epoch {epoch + 1}: Train Acc: {accuracy:.2f}%, Val Acc: {validation_accuracy:.2f}%, Val F1: {val_f1:.4f}")

    if verbose:
        print(f'Finished training, time taken: {time.time() - train_start:.2f}s')
        print(f'Best Val Accuracy: {highest_acc:.2f}%, Best Val F1: {highest_f1:.4f}')

    return cnn, running_record, model_history



class_weights = torch.tensor([1 / 136, 1 / 1104, 1 / 201, 1 / 206], dtype=torch.float).cuda()
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
k = 3
batch_size = 16
lr_factor = 0.6  #0.5
f1_value = 0.6

X_train_val = train_val_df.drop(columns=['label'])
y_train_val = train_val_df['label']
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
cv_results = []

for fold, (train_index, val_index) in enumerate(kf.split(X_train_val, y_train_val)):
    print(f'Cross validation: Fold {fold}')
    train_df = train_val_df.iloc[train_index]
    val_df = train_val_df.iloc[val_index]

    # Load EfficientNet-B0 pretrained
    # weights = EfficientNet_B0_Weights.DEFAULT
    # net = efficientnet_b0(weights=weights)
    # net.classifier[1] = nn.Linear(net.classifier[1].in_features, 4)

    # MobileNetV2
    # weights = MobileNet_V2_Weights.DEFAULT
    # net = mobilenet_v2(weights=weights)
    # net.classifier[1] = nn.Linear(net.classifier[1].in_features, 4)

    #ResNet50
    weights = ResNet50_Weights.DEFAULT
    net = resnet50(weights=weights)
    net.fc = nn.Linear(net.fc.in_features, 4)

    net = net.to(device)
    model_name = net.__class__.__name__

    opt = optim.Adam(net.parameters(), lr=0.001 * lr_factor, weight_decay=1e-4)

    train_ds = RGBConcatImageDataset(train_df, transforms=train_transform)
    val_ds = RGBConcatImageDataset(val_df, transforms=test_transform)
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    today = datetime.datetime.now().strftime("%d-%m_%H%M")
    _, results, history = train(cnn=net, epochs=5,  #20,
                                optimizer=opt, loss_fn=loss_fn,
                                train_loader=train_dataloader, class_weights=class_weights,
                                lr_scheduler=None,
                                valid_loader=valid_dataloader,
                                device=device,
                                set_highest_f1=f1_value,
                                current_date=today, save=True, verbose=True)

    plot_accuracy(history, f'Fold {fold}')
    cv_results.append(results)

cv_df = pd.DataFrame.from_records(cv_results)
cv_df.to_csv(f'{model_name}_cv_pretrained.csv', index=False)
# cv_df.to_excel(f'{model_name}_cv_pretrained.xlsx', index=False)
print('Cross validation results saved to cv_pretrained.csv')
print(cv_df)

print('Total runtime:', time.time() - st)
