import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, roc_auc_score, \
    recall_score
import cv2
import matplotlib.pyplot as plt

from cnn import ImageDataset
from torch.utils.data import DataLoader
import datetime
import time
import numpy as np
from misc_utils import set_seed, get_current_time, check_predictions

st = time.time()
set_seed(42)
img_directory = '../dataset/prepared'


def test_model(cnn, loader, loss_fn, class_weights,
               device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    test_correct, test_total = 0, 0
    test_running_loss = 0.0
    class_correct, class_total = [0, 0, 0], [0, 0, 0]
    all_pred, all_labels, all_probs = [], [], []
    results = {'accuracy': None,
               'f1': None,
               'precision': None,
               'recall': None,
               'roc-auc': None,
               'class_weights': class_weights,
               'loss': None,
               'all_pred': None,
               'all_labels': None,
               'all_probs': None
               }
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = cnn(inputs)
            prob = F.softmax(outputs, dim=1)
            _, pred = torch.max(outputs, 1)
            test_correct += (pred == labels).sum().item()
            test_total += labels.size(0)

            # Weighted Accuracy
            for i in range(3):
                class_idx = (labels == i)
                class_correct[i] += (pred[class_idx] == i).sum().item()
                class_total[i] += class_idx.sum().item()

            all_pred.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(prob.cpu().numpy())

            loss = loss_fn(outputs, labels)
            test_running_loss += loss.item()

    class_accuracies = [correct / total if total > 0 else 0 for correct, total in zip(class_correct, class_total)]
    weighted_accuracy = sum(weight * ca for weight, ca in zip(class_weights, class_accuracies)) / class_weights.sum()
    f1_2 = f1_score(all_labels, all_pred, labels=[2], average='macro')
    f1_1 = f1_score(all_labels, all_pred, labels=[1], average='macro')

    results['accuracy'] = round(weighted_accuracy.item()*100, 4)
    results['f1'] = round(f1_score(all_labels, all_pred, average='macro'), 4)
    results['roc-auc'] = round(roc_auc_score(all_labels, all_probs, multi_class='ovo'), 4)
    results['precision'] = round(precision_score(all_labels, all_pred, average='macro'), 4)
    results['recall'] = round(recall_score(all_labels, all_pred, average='macro'), 4)
    results['all_pred'] = all_pred
    results['all_labels'] = all_labels
    results['all_probs'] = all_probs
    results['loss'] = round(test_running_loss, 4)

    conf_matrix = confusion_matrix(all_labels, all_pred, labels=[0, 1, 2])
    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['particles', 'single', 'multi'])
    disp.plot()
    plt.show()

    test_acc = (test_correct / test_total) * 100
    print(f'Test accuracy:{test_acc:.4f}')
    print(f"Weighted Accuracy: {weighted_accuracy*100:.4f}, F1 Score('Multi'): {f1_2:.4f}, F1 Score('Single'): {f1_1:.4f}")
    print(f"roc-auc: {results['roc-auc']:.4f}, precision: {results['precision']:.4f}, recall: {results['recall']:.4f}")
    print(f"F1 Score ('Average'): {results['f1']:.4f}, loss: {results['loss']:.4f}")

    return results


test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.6733, 0.6733, 0.6733), (0.0598, 0.0598, 0.0598))
])
test_ds = ImageDataset(img_dir=img_directory, mode='test', transforms=test_transform)
batch_size = 16
test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

class_weights = [0.2, 0.3, 0.5]  # ['particles', 'single', 'multi']
class_names = ['particles', 'single', 'multi']
class_weights = torch.FloatTensor(class_weights).cuda()
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = f'./saved_models/22102024_1217_90_model.pt'
model = torch.load(model_path, weights_only=False)
model.eval()
results = test_model(model, loader=test_dataloader,
                     class_weights=class_weights,
                     loss_fn=loss_fn, device=device)

image_list = test_ds.get_image_list()

check_predictions(image_list, predictions=results['all_pred'],
                  labels=results['all_labels'],
                  lim=10,
                  class_names=class_names,
                  save=True)

results_df = pd.DataFrame([results])
results_df.to_excel('test_results_multicellclassification.xlsx', index=False)

print('Time taken: ', time.time() - st)
