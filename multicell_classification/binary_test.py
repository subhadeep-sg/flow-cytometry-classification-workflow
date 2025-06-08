import os
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
img_directory = './classify_dataset/test'

train_dir = './classify_dataset/train'
test_dir = './classify_dataset/test'


def count_images(folder):
    return sum([len(files) for _, _, files in os.walk(folder)])


print(f"Number of training images: {count_images(train_dir)}")
print(f"Number of validation images: {count_images(test_dir)}")


def test_model(cnn, loader, loss_fn, class_weights,
               device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), plot=True):
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

    if plot:
        plt.figure()
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Cluster', 'Non Cluster'])
        disp.plot()
        plt.tight_layout()
        plt.show()

    test_acc = (test_correct / test_total) * 100
    print(f'Test accuracy:{test_acc:.4f}')
    print(
        f"Weighted Accuracy: {weighted_accuracy * 100:.4f}, F1 Score('Non Cluster'): {f1_2:.4f}, F1 Score('Cluster'): {f1_1:.4f}")
    print(f"roc-auc: {results['roc-auc']:.4f}, precision: {results['precision']:.4f}, recall: {results['recall']:.4f}")
    print(f"F1 Score ('Average'): {results['f1']:.4f}, loss: {results['loss']:.4f}")

    return results


test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.6558058, 0.6558058, 0.6558058), (0.11742277, 0.11742277, 0.11742277))
])

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

# model_path = f'./saved_models/19-01_2146_90_model.pt'
model_path = f'./saved_models/04-04_1836_89_model.pt'
model = torch.load(model_path, weights_only=False)
model.eval()
results = test_model(model, loader=test_dataloader,
                     class_weights=class_weights,
                     loss_fn=loss_fn, device=device)

image_list = test_ds.get_image_list()


# revised_class_names = ['cluster', 'non_cluster']
# check_predictions(image_list, predictions=results['all_pred'],
#                   labels=results['all_labels'],
#                   lim=10,
#                   class_names=revised_class_names,
#                   save=False)   # True


def misclassification_saving(df):
    preds = df['all_pred']
    gt = df['all_labels']

    img_dir = '../multicell_classification/classify_dataset/test'
    image_list, labels = [], []
    for dir_path, _, filenames in os.walk(img_dir):
        for filename in filenames:
            image_list.append(os.path.join(dir_path, filename))
            labels.append(dir_path.split('\\')[-1])

    relevant_indices = [i for i in range(len(preds)) if preds[i] != gt[i]]

    selected_misses = [image_list[i] for i in relevant_indices[0:5]]
    actual_labels = [gt[i] for i in relevant_indices[0:5]]
    predictions = [preds[i] for i in relevant_indices[0:5]]
    images = [cv2.resize(cv2.imread(img), (224, 224)) for img in selected_misses]

    label_names = ['cluster', 'non cluster']
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    axes = axes.ravel()
    for i, ax in enumerate(axes):
        ax.imshow(images[i])
        ax.set_title(f"Label {label_names[actual_labels[i]]}, \n Predicted {label_names[predictions[i]]}")
        ax.axis("off")
    # fig.suptitle(f"{df['Model Name']}")
    plt.tight_layout()
    plt.show()

    print(f"Model Name: Simple CNN")
    print(len(relevant_indices))
    print(relevant_indices)
    print(selected_misses)

    print('---------')


# misclassification_saving(results)
results_df = pd.DataFrame([results])
# results_df.to_excel('test_results_binary_classification_4april.xlsx', index=False)

print('Time taken: ', time.time() - st)
