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
from multicell_classification.cnn import ConvNet, ImageDataset
import torch.optim as optim
import torch.nn as nn
import time
import datetime
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, auc, precision_score, \
    recall_score
from multicell_classification.misc_utils import set_seed, get_current_time, plot_accuracy, mean_std_computation, \
    check_predictions
from torchvision.models import resnet50, ResNet50_Weights, mobilenet_v2, efficientnet_b0, vgg16, densenet121
from torchvision.io import decode_image
from torchvision.models import EfficientNet_B0_Weights, MobileNet_V2_Weights, DenseNet121_Weights, VGG16_Weights


st = time.time()
set_seed(42)
img_directory = '../multicell_classification/classify_dataset/train'

image_list, labels = [], []
for dir_path, _, filenames in os.walk(img_directory):
    for filename in filenames:
        image_list.append(os.path.join(dir_path, filename))
        labels.append(dir_path.split('\\')[-1])

df = pd.DataFrame(columns=['filename', 'class_label'])
df['filename'] = pd.Series(image_list)
df['class_label'] = pd.Series(labels)

label_list = df['class_label'].tolist()
label_mapping = {'cluster_revised': 0, 'non_cluster_revised': 1}
label_list = [label_mapping[label] for label in label_list]

train_val_df = df
print(f'Training {len(train_val_df)}')
print(train_val_df['class_label'].value_counts())

mean, std = mean_std_computation(img_df=train_val_df)
print(f'Mean and Std for normalizing: {mean, std}')

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((mean[0], mean[1], mean[2]), (std[0], std[1], std[2]))
])
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((mean[0], mean[1], mean[2]), (std[0], std[1], std[2]))
])


def train(cnn, epochs, class_weights, train_loader, valid_loader, optimizer, lr_scheduler, loss_fn,
          save=False, current_date=None,
          set_highest_acc=70.0, set_highest_f1=0.7, device=torch.device('cuda'), verbose=False):
    if verbose:
        print('Training start....')
    train_start = time.time()
    highest_acc = set_highest_acc
    highest_f1 = set_highest_f1
    model_history = {'train': [], 'valid': []}
    running_record = {'f1': set_highest_f1,
                      'accuracy': set_highest_acc,
                      'roc-auc': None,
                      'recall': None,
                      'precision': None,
                      'loss': None,
                      'class_weights': class_weights,
                      'optimizer': optimizer,
                      'model path': None
                      }
    for epoch in range(epochs):
        running_loss, val_running_loss = 0.0, 0.0
        total_samples, val_total = 0, 0
        pred_correct, val_pred_correct = 0, 0

        class_correct, class_total = [0, 0, 0], [0, 0, 0]
        val_class_correct, val_class_total = [0, 0, 0], [0, 0, 0]

        all_pred, all_labels, all_probs = [], [], []
        val_pred, val_labels, val_probs = [], [], []

        for inputs, labels in train_loader:
            # print(inputs.numpy().shape)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = cnn(inputs)
            probs = F.softmax(outputs, dim=1)
            _, pred = torch.max(outputs, 1)
            # Update the running total of correct predictions and samples
            pred_correct += (pred == labels).sum().item()

            # Finding accuracy in predicting class 2 ie. 'multi'
            for i in range(3):
                class_idx = (labels == i)
                class_correct[i] += (pred[class_idx] == i).sum().item()
                class_total[i] += class_idx.sum().item()

            all_pred.append(pred.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())

            # -----------------------
            total_samples += labels.size(0)
            # print(outputs.shape)

            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()
            if lr_scheduler:
                lr_scheduler.step()

            # print statistics
            running_loss += loss.item()

        all_pred = np.concatenate(all_pred, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        cnn.eval()
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = cnn(inputs)
                prob = F.softmax(outputs, dim=1)
                _, pred = torch.max(outputs, 1)

                # Overall accuracy
                val_pred_correct += (pred == labels).sum().item()
                val_total += labels.size(0)

                # Weighted accuracy for validation
                for i in range(3):
                    class_idx = (labels == i)
                    val_class_correct[i] += (pred[class_idx] == i).sum().item()
                    val_class_total[i] += class_idx.sum().item()

                val_pred.append(pred.cpu().numpy())
                val_labels.append(labels.cpu().numpy())
                val_probs.append(prob.detach().cpu().numpy())

                # Loss
                validation_loss = loss_fn(outputs, labels)
                val_running_loss += validation_loss.item()

        cnn.train()
        val_pred = np.concatenate(val_pred, axis=0)
        val_probs = np.concatenate(val_probs, axis=0)

        val_labels = np.concatenate(val_labels, axis=0)

        # Calculating accuracy
        validation_accuracy = (val_pred_correct / val_total) * 100
        accuracy = (pred_correct / total_samples) * 100

        # Calculating weighted accuracy and f1-score for training
        class_accuracies = [correct / total if total > 0 else 0 for correct, total in zip(class_correct, class_total)]
        weighted_accuracy = sum(
            weight * ca for weight, ca in zip(class_weights, class_accuracies)) / class_weights.sum()

        # Calculating weighted accuracy, f1-score for validation
        val_class_accuracies = [correct / total if total > 0 else 0 for correct, total in
                                zip(val_class_correct, val_class_total)]
        val_weighted_accuracy = sum(
            weight * ca for weight, ca in zip(class_weights, val_class_accuracies)) / class_weights.sum()
        val_f1_2 = f1_score(val_labels, val_pred, labels=[1], average='macro')
        val_f1_1 = f1_score(val_labels, val_pred, labels=[0], average='macro')

        val_f1 = f1_score(val_labels, val_pred, average='macro')
        val_roc_auc = roc_auc_score(val_labels, val_probs[:, 1], multi_class='ovo')
        val_precision = precision_score(val_labels, val_pred, average='macro')
        val_recall = recall_score(val_labels, val_pred, average='macro')

        # Adding accuracy values for plotting accuracy curve
        model_history['train'].append(accuracy)
        model_history['valid'].append(validation_accuracy)

        # if validation_accuracy > highest_acc and val_f1_2 > highest_f1:
        if val_f1_1 > highest_f1:
            highest_acc = val_weighted_accuracy.item()  # validation_accuracy
            highest_f1 = val_f1_1
            conf_matrix = confusion_matrix(val_labels, val_pred, labels=[0, 1])
            running_record['f1'] = round(val_f1, 4)
            running_record['accuracy'] = round(validation_accuracy, 4)
            running_record['precision'] = round(val_precision, 4)
            running_record['recall'] = round(val_recall, 4)
            running_record['roc-auc'] = round(val_roc_auc, 4)
            running_record['loss'] = round(val_running_loss, 4)
            if save:
                os.makedirs('saved_models', exist_ok=True)
                running_record[
                    'model path'] = f'./saved_models/{model_name}_{current_date}_{round(highest_acc * 100)}_model.pt'
                torch.save(net, f'./saved_models/{model_name}_{current_date}_{round(highest_acc * 100)}_model.pt')

        if verbose:
            print(f'Epoch:{epoch}, running loss:{running_loss:.4f}, train accuracy:{accuracy:.4f}, '
                  f'validation loss:{val_running_loss:.4f}, valid accuracy:{validation_accuracy:.4f}')
            print(
                f"Validation: Weighted Accuracy: {val_weighted_accuracy:.4f}, F1 Score('Non Cluster'): {val_f1_2:.4f}, "
                f"F1 Score('Cluster'): {val_f1_1:.4f}")

    if verbose:
        print(f'Finished training, time taken:{time.time() - train_start}')

    # if (highest_acc > set_highest_acc and highest_f1 > set_highest_f1) and verbose:
    if highest_f1 > set_highest_f1 and verbose:
        print(f'Highest validation weighted accuracy obtained: ', highest_acc)
        print(f'Highest F1 score obtained: ', highest_f1)
        print(f'Model saved at ./saved_models/{model_name}_{current_date}_{round(highest_acc * 100)}_model.pt')

    return cnn, running_record, model_history


class_weights = [0.5, 0.5]
class_weights = torch.FloatTensor(class_weights).cuda()
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

k = 3
batch_size = 16
y_train_val = train_val_df['class_label']
X_train_val = train_val_df.drop(columns=['class_label'])
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
cv_results = []
lr_factor = 0.5  # 0.9
f1_value = 0.87
# #
for fold, (train_index, val_index) in enumerate(kf.split(X_train_val, y_train_val)):
    print(f'Cross validation: Fold {fold}')
    train_df = train_val_df.iloc[train_index]
    val_df = train_val_df.iloc[val_index]

    # Resnet50
    # weights = ResNet50_Weights.DEFAULT
    # net = resnet50(weights=weights)
    # net.fc = nn.Linear(net.fc.in_features, 2)
    #
    # EfficientNet-B0
    # weights = EfficientNet_B0_Weights.DEFAULT
    # net = efficientnet_b0(weights=weights)
    # net.classifier[1] = nn.Linear(net.classifier[1].in_features, 2)

    # MobileNetV2
    # weights = MobileNet_V2_Weights.DEFAULT
    # net = mobilenet_v2(weights=weights)
    # net.classifier[1] = nn.Linear(net.classifier[1].in_features, 2)

    # DenseNet-121
    # weights = DenseNet121_Weights.DEFAULT
    # net = densenet121(weights=weights)
    # net.classifier = nn.Linear(net.classifier.in_features, 2)

    # VGG-16
    weights = VGG16_Weights.DEFAULT
    net = vgg16(weights=weights)
    net.classifier[6] = nn.Linear(net.classifier[6].in_features, 2)

    net = net.to(device)
    model_name = net.__class__.__name__

    opt = optim.Adam(net.parameters(), lr=0.001 * lr_factor, weight_decay=1e-4)

    train_ds = ImageDataset(dataframe=train_df, mode='train', transforms=transform)
    val_ds = ImageDataset(dataframe=val_df, mode='val', transforms=test_transform)
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    today = datetime.datetime.now().strftime("%d-%m_%H%M")
    _, results, history = train(cnn=net, epochs=20,  # 80
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
cv_df.to_excel(f'{model_name}_cv_pretrained.xlsx', index=False)
print('Cross validation results saved to cv_pretrained.csv')
print(cv_df)

print('Total runtime:', time.time() - st)
