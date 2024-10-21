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
# from color_classification.rgb_test import test_rgb_model
from rgb_cnn import RGBConvNet, RGBImageDataset, normalization_channel_wise
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, precision_score, recall_score, roc_curve, auc
from multicell_classification.misc_utils import plot_accuracy, set_seed, plot_roc

set_seed(42)
st = time.time()

df = pd.read_csv('groundtruths.csv')

label_list = df['label'].tolist()
label_mapping = {'rbc': 0, 'platelet': 1, 'wbc platelet': 2, 'wbc': 3}
label_list = [label_mapping[label] for label in label_list]

test_size = 0.3
valid_split = 0.2
train_size = 1 - test_size
valid_size = valid_split / train_size
train_val_df, test_df = train_test_split(df, test_size=test_size, stratify=df['label'], random_state=42)
# train_df, val_df = train_test_split(train_val_df, test_size=valid_size, stratify=train_val_df['label'], random_state=42)

# normalization_channel_wise(train_df)
# print(f'Train valid test split {len(train_df)}, {len(val_df)}, {len(test_df)}')
print(f'Training {len(train_val_df)}, Test {len(test_df)}')

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
])
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])


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
        class_correct, class_total = [0, 0, 0], [0, 0, 0]
        val_class_correct, val_class_total = [0, 0, 0], [0, 0, 0]
        all_pred, all_labels = [], []
        val_pred, val_labels = [], []
        all_probs, val_probs = [], []
        for inputs, labels in train_loader:
            inputs[0] = inputs[0].to(device)
            inputs[1] = inputs[1].to(device)
            inputs[2] = inputs[2].to(device)
            inputs[3] = inputs[3].to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs[0], inputs[1], inputs[2], inputs[3])
            probs = F.softmax(outputs, dim=1)
            # print(outputs.size())
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
        for inputs, labels in valid_loader:
            inputs[0] = inputs[0].to(device)
            inputs[1] = inputs[1].to(device)
            inputs[2] = inputs[2].to(device)
            inputs[3] = inputs[3].to(device)
            labels = labels.to(device)

            outputs = net(inputs[0], inputs[1], inputs[2], inputs[3])
            probs = F.softmax(outputs, dim=1)
            _, pred = torch.max(outputs, 1)

            val_pred_correct += (pred == labels).sum().item()

            val_total += labels.size(0)

            # for i in range(3):
            #     class_idx = (labels == i)
            #     val_class_correct[i] += (pred[class_idx] == i).sum().item()
            #     val_class_total[i] += class_idx.sum().item()

            val_pred.append(pred.cpu().numpy())
            val_probs.append(probs.detach().cpu().numpy())
            val_labels.append(labels.cpu().numpy())

            validation_loss = loss_fn(outputs, labels)
            val_running_loss += validation_loss.item()

        val_pred = np.concatenate(val_pred, axis=0)
        val_probs = np.concatenate(val_probs, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)

        accuracy = (pred_correct / total_samples) * 100
        validation_accuracy = (val_pred_correct / val_total) * 100

        # f1_2 = f1_score(all_labels, all_pred, labels=[2], average='macro')
        # f1_1 = f1_score(all_labels, all_pred, labels=[1], average='macro')
        # f1_0 = f1_score(all_labels, all_pred, labels=[0], average='macro')
        roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovo')

        # val_f1_2 = f1_score(val_labels, val_pred, labels=[2], average='macro')
        # val_f1_1 = f1_score(val_labels, val_pred, labels=[1], average='macro')
        # val_f1_0 = f1_score(all_labels, all_pred, labels=[0], average='macro')
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
            final_res['f1'] = round(val_f1, 4)
            final_res['precision'] = round(val_precision, 4)
            final_res['recall'] = round(val_recall, 4)
            final_res['accuracy'] = round(validation_accuracy, 4)
            final_res['loss'] = round(val_running_loss, 5)
            final_res['all_pred'] = val_pred
            final_res['all_labels'] = val_labels
            final_res['all_probs'] = val_probs
            if save:
                os.makedirs('saved_rgb_models', exist_ok=True)

                torch.save(net, f'./saved_rgb_models/{today}_{round(val_roc_auc * 100)}_model.pt')
                final_path = f'./saved_rgb_models/{today}_{round(val_roc_auc * 100)}_model.pt'
                final_res['model path'] = final_path

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


# batch_size = 16
# loss_fn = nn.CrossEntropyLoss()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# rgbnet = RGBConvNet(num_channels=3, device=device, batch_size=batch_size).to(device)
# lr_factor = 0.9
# opt = Adam(rgbnet.parameters(), lr=0.001 * lr_factor, weight_decay=1e-4)
# set_high_acc = 85.0
# set_high_f1 = 0.86
# roc_auc_val = 0.88
# rgb_model, highest_acc, valid_acc, history = rgb_train(net=rgbnet, epochs=20, optimizer=opt, loss_fn=loss_fn,
#                                                        train_loader=train_dataloader,
#                                                        valid_loader=valid_dataloader,
#                                                        device=device,
#                                                        set_roc_auc=roc_auc_val,
#                                                        current_date=today, save=False, verbose=True)
# #
# plot_accuracy(history)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
k = 3
batch_size = 16
y_train_val = train_val_df['label']
X_train_val = train_val_df.drop(columns=['label'])
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
cv_results = []

class_weights = torch.tensor([1/136, 1/1104, 1/201, 1/206], dtype=torch.float).to(device)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

lr_factor = 0.9
roc_auc_val = 0.9
# Start cross-validation
for fold, (train_index, val_index) in enumerate(kf.split(X_train_val, y_train_val)):
    print(f'Cross Validating: Fold {fold}')
    train_df = train_val_df.iloc[train_index]
    val_df = train_val_df.iloc[val_index]
    # print(f'Train set:\n{train_df}\n')
    # print(f'Valid set:\n{val_df}\n')

    # New model initialization every fold
    rgbnet = RGBConvNet(num_channels=3, device=device, batch_size=batch_size).to(device)
    opt = Adam(rgbnet.parameters(), lr=0.001 * lr_factor, weight_decay=1e-4)

    train_ds = RGBImageDataset(dataframe=train_df, mode='train', transforms=transform)
    val_ds = RGBImageDataset(dataframe=val_df, mode='val', transforms=test_transform)
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
    today = datetime.datetime.now().strftime("%d-%m_%H%M")
    _, results, history = rgb_train(net=rgbnet, epochs=50, optimizer=opt, loss_fn=loss_fn,
                                    train_loader=train_dataloader,
                                    valid_loader=valid_dataloader,
                                    device=device,
                                    set_roc_auc=roc_auc_val,
                                    class_weights=class_weights,
                                    current_date=today, save=True, verbose=True)
    plot_accuracy(history, f'Fold {fold}')
    plot_roc(results, ['rbc', 'platelet', 'wbc platelet', 'wbc'])
    cv_results.append(results)

cv_df = pd.DataFrame.from_records(cv_results)
cv_df.to_csv('cross_validation_results.csv', index=False)
print('Cross validation results saved to cross_validation_results.csv')

# batch_size = 16
# test_ds = RGBImageDataset(dataframe=test_df, mode='test', transforms=test_transform)
# test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)
# loss_fn = nn.CrossEntropyLoss()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# model_path = './saved_rgb_models/14-10_1108_99_model.pt'
# model = torch.load(model_path, weights_only=False)
# model.eval()
# res = test_rgb_model(net=model, loader=test_loader,
#                      loss_fn=loss_fn, device=device)
#
# print(res)
# res_df = pd.DataFrame([res])
# print(res_df)
#
# # res_df.to_csv('test_set_results.csv', index=False)

print('Total time taken: ', time.time() - st)
