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
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.io import decode_image

st = time.time()
set_seed(42)
img_directory = '../multicell_classification/classify_dataset/test'

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.6558058, 0.6558058, 0.6558058), (0.11742277, 0.11742277, 0.11742277))
])


def test_model(cnn, loader, loss_fn, class_weights,
               device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    test_correct, test_total = 0, 0
    test_running_loss = 0.0
    class_correct, class_total = [0, 0, 0], [0, 0, 0]
    all_pred, all_labels, all_probs = [], [], []
    results = {'accuracy': None,
               'f1': None,
               'roc-auc': None,
               'precision': None,
               'recall': None,
               'loss': None,
               'class_weights': class_weights,
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
            all_probs.append(prob.cpu().numpy())

            loss = loss_fn(outputs, labels)
            test_running_loss += loss.item()

    all_probs = np.concatenate(all_probs, axis=0)

    class_accuracies = [correct / total if total > 0 else 0 for correct, total in zip(class_correct, class_total)]
    weighted_accuracy = sum(weight * ca for weight, ca in zip(class_weights, class_accuracies)) / class_weights.sum()
    f1_2 = f1_score(all_labels, all_pred, labels=[1], average='macro')
    f1_1 = f1_score(all_labels, all_pred, labels=[0], average='macro')

    results['accuracy'] = round(weighted_accuracy.item() * 100, 4)
    results['f1'] = round(f1_score(all_labels, all_pred, average='macro'), 4)
    results['roc-auc'] = round(roc_auc_score(all_labels, all_probs[:, 1], multi_class='ovo'), 4)
    results['precision'] = round(precision_score(all_labels, all_pred, average='macro'), 4)
    results['recall'] = round(recall_score(all_labels, all_pred, average='macro'), 4)
    results['all_pred'] = all_pred
    results['all_labels'] = all_labels
    results['all_probs'] = all_probs
    results['loss'] = round(test_running_loss, 4)

    conf_matrix = confusion_matrix(all_labels, all_pred, labels=[0, 1])
    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Cluster', 'Non Cluster'])
    # disp.plot()
    # plt.tight_layout()
    # plt.show()

    test_acc = (test_correct / test_total) * 100
    print(f'Test accuracy:{test_acc:.4f}')
    print(
        f"Weighted Accuracy: {weighted_accuracy * 100:.4f}, F1 Score('Non Cluster'): {f1_2:.4f}, F1 Score('Cluster'): {f1_1:.4f}")
    print(f"roc-auc: {results['roc-auc']:.4f}, precision: {results['precision']:.4f}, recall: {results['recall']:.4f}")
    print(f"F1 Score ('Average'): {results['f1']:.4f}, loss: {results['loss']:.4f}")

    return results


image_list, labels = [], []
for dir_path, _, filenames in os.walk(img_directory):
    for filename in filenames:
        image_list.append(os.path.join(dir_path, filename))
        labels.append(dir_path.split('\\')[-1])

test_df = pd.DataFrame(columns=['filename', 'class_label'])
test_df['filename'] = pd.Series(image_list)
test_df['class_label'] = pd.Series(labels)

label_list = test_df['class_label'].tolist()
label_mapping = {'cluster_revised': 0, 'non_cluster_revised': 1}
label_list = [label_mapping[label] for label in label_list]

test_ds = ImageDataset(dataframe=test_df, mode='test', transforms=test_transform)
batch_size = 16
test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

class_weights = [0.5, 0.5]  # ['particles', 'single', 'multi']
class_names = ['cluster_revised', 'non_cluster_revised']
class_weights = torch.FloatTensor(class_weights).cuda()
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# weights = ResNet50_Weights.DEFAULT
# model = resnet50(weights=weights)
# model.fc = nn.Linear(model.fc.in_features, 2)
# model = model.to(device)
# model.eval()

# Implementing all the best pretrained models comparison
pretrained_models = [
    './saved_models/ResNet_06-02_0524_91_model.pt',
    './saved_models/EfficientNet_06-02_0735_93_model.pt',
    './saved_models/MobileNetV2_06-02_0749_92_model.pt',
    './saved_models/DenseNet_06-02_0901_92_model.pt'
]

final_results = pd.DataFrame()
for model_name in pretrained_models:
    model = torch.load(model_name, weights_only=False)
    model.eval()
    results = test_model(model, loader=test_dataloader,
                         class_weights=class_weights,
                         loss_fn=loss_fn, device=device)
    results_df = pd.DataFrame([results])
    final_results = pd.concat([final_results, results_df], ignore_index=True)
    # final_results.loc[len(final_results)] = [results]

model_names = ['ResNet', 'EfficientNet', 'MobileNetV2', 'DenseNet']
final_results.insert(loc=0, column='Model Name', value=model_names)
# print(final_results)


# final_results.to_excel('pretrained_test_results.xlsx', index=False)


# model_path = f'./saved_models/ResNet_06-02_0524_91_model.pt'
# model = torch.load(model_path, weights_only=False)
# model.eval()

# results = test_model(model, loader=test_dataloader,
#                      class_weights=class_weights,
#                      loss_fn=loss_fn, device=device)

# print(results)

# image_list = test_ds.get_image_list()
#
# results_df = pd.DataFrame([results])
# results_df.to_excel('test_resnet.xlsx', index=False)

# revised_class_names = ['cluster', 'non_cluster']
# check_predictions(image_list, predictions=results['all_pred'],
#                   labels=results['all_labels'],
#                   lim=10,
#                   class_names=revised_class_names,
#                   save=False)

def misclassification_saving(df):
    preds = df['all_pred']
    gt = df['all_labels']

    img_dir = '../multicell_classification/classify_dataset/test'
    image_list, labels = [], []
    for dir_path, _, filenames in os.walk(img_dir):
        for filename in filenames:
            image_list.append(os.path.join(dir_path, filename))
            labels.append(dir_path.split('\\')[-1])

    for k in range(len(df['Model Name'])):
        relevant_indices = [i for i in range(len(preds[k])) if preds[k][i] != gt[k][i]]

        selected_misses = [image_list[i] for i in relevant_indices[0:5]]
        actual_labels = [gt[k][i] for i in relevant_indices[0:5]]
        predictions = [preds[k][i] for i in relevant_indices[0:5]]
        images = [cv2.resize(cv2.imread(img), (224, 224)) for img in selected_misses]

        label_names = ['cluster', 'non cluster']
        fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
        axes = axes.ravel()
        for i, ax in enumerate(axes):
            ax.imshow(images[i])
            ax.set_title(f"Label {label_names[actual_labels[i]]}, \n Predicted {label_names[predictions[i]]}")
            ax.axis("off")
        # fig.suptitle(f"{df['Model Name'][k]}")
        plt.tight_layout()
        plt.show()

        print(f"Model Name: {df['Model Name'][k]}")
        print(len(relevant_indices))
        print(relevant_indices)
        print(selected_misses)

        print('---------')



misclassification_saving(final_results)

print('Total runtime:', time.time() - st)
