import torch
import numpy as np
import pandas as pd
import time
import cv2
import os
import datetime
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from color_classification.rgb_cnn import RGBImageDataset
import torch.nn.functional as F
from multicell_classification.misc_utils import set_seed
from sklearn.metrics import f1_score, roc_auc_score, ConfusionMatrixDisplay, confusion_matrix, precision_score, \
    recall_score, roc_curve, auc

st = time.time()
set_seed(42)


def test_rgb_model(net, loader, loss_fn, device=torch.device('cuda')):
    test_correct, test_total = 0, 0
    test_running_loss = 0.0
    all_pred, all_labels, all_probs = [], [], []
    all_2, all_3, all_7, all_11 = [], [], [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs[0] = inputs[0].to(device)
            inputs[1] = inputs[1].to(device)
            inputs[2] = inputs[2].to(device)
            inputs[3] = inputs[3].to(device)
            labels = labels.to(device)

            outputs = net(inputs[0], inputs[1], inputs[2], inputs[3])
            _, pred = torch.max(outputs, 1)
            probs = F.softmax(outputs, dim=1)

            test_correct += (pred == labels).sum().item()
            test_total += labels.size(0)

            all_pred.append(pred.cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_2.append(inputs[0].cpu().numpy())
            all_3.append(inputs[1].cpu().numpy())
            all_7.append(inputs[2].cpu().numpy())
            all_11.append(inputs[3].cpu().numpy())

            loss = loss_fn(outputs, labels)
            test_running_loss += loss.item()

    all_pred = np.concatenate(all_pred, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    accuracy = (test_correct / test_total) * 100

    f1_3 = f1_score(all_labels, all_pred, labels=[3], average='macro')
    f1_2 = f1_score(all_labels, all_pred, labels=[2], average='macro')
    f1_1 = f1_score(all_labels, all_pred, labels=[1], average='macro')
    f1_0 = f1_score(all_labels, all_pred, labels=[0], average='macro')

    f1 = f1_score(all_labels, all_pred, average='macro')
    roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovo')
    precision = precision_score(all_labels, all_pred, average='macro')
    recall = recall_score(all_labels, all_pred, average='macro')

    class_names = ['rbc', 'platelet', 'wbc platelet', 'wbc']
    for i in range(3):
        fpr, tpr, thresh = roc_curve(all_labels, all_probs[:, i], pos_label=i)
        auc_score = auc(fpr, tpr)
        plt.title(f'Receiver Operating Characteristic for class {class_names[i]}')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % auc_score)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

    conf_matrix = confusion_matrix(all_labels, all_pred, labels=[0, 1, 2, 3])
    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['rbc', 'platelet', 'wbc platelet', 'wbc'])
    disp.plot()
    plt.tight_layout()
    plt.show()

    results = {'f1': round(f1, 4),
               'f1 rbc': round(f1_0, 4),
               'f1 platelet': round(f1_1, 4),
               'f1 wbc platelet': round(f1_2, 4),
               'f1 wbc': round(f1_3, 4),
               'roc-auc': round(roc_auc, 4),
               'accuracy': round(accuracy, 4),
               'precision': round(precision, 4),
               'recall': round(recall, 4),
               'cross-entropy loss': round(test_running_loss, 4),
               'all_labels': all_labels,
               'all_pred': all_pred,
               '2': all_2,
               '3': all_3,
               '7': all_7,
               '11': all_11
               }

    return results


today = datetime.datetime.now().strftime("%d-%m_%H")
df = pd.read_csv('groundtruths.csv')

label_list = df['label'].tolist()
label_mapping = {'rbc': 0, 'platelet': 1, 'wbc platelet': 2, 'wbc': 3}
label_list = [label_mapping[label] for label in label_list]

test_size = 0.3
train_val_df, test_df = train_test_split(df, test_size=test_size, stratify=df['label'], random_state=42)
batch_size = 1
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])
test_ds = RGBImageDataset(dataframe=test_df, mode='test', transforms=test_transform)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_weights = torch.tensor([1/136, 1/1104, 1/201, 1/206], dtype=torch.float).to(device)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

model_path = './saved_rgb_models/19-10_1733_98_model.pt'
model = torch.load(model_path, weights_only=False)
model.eval()
res = test_rgb_model(net=model, loader=test_loader,
                     loss_fn=loss_fn, device=device)

gts = res['all_labels'].tolist()
preds = res['all_pred'].tolist()
print(res['f1'])
print(res['accuracy'])
channel2s = res['2']
channel3s = res['3']
channel7s = res['7']
channel11s = res['11']

class_names = ['rbc', 'platelet', 'wbc platelet', 'wbc']


def plot_all_image_channels(images, main_title="All Channel Images", save=False, save_name=None):
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle(main_title, fontsize=20)

    for i, ax in enumerate(axes):
        ax.imshow(images[i])
        ax.axis('off')
    plt.show()

    if save:
        assert save_name is not None, "Please provide figure name"
        os.makedirs('./p2_misclassifications', exist_ok=True)
        fig.savefig(f'./p2_misclassifications/{save_name}')
        plt.close(fig)


def save_misclassification(predictions, groundtruths, n, class_names, df=None, file_list=None):
    assert len(df) == len(predictions) == len(groundtruths), "Predictions and Image List aren't same size!"
    # print(f'Length of predictions and groundtruths: {len(predictions)}, {len(groundtruths)}')
    misclassified = []
    for i in range(len(predictions)):
        if predictions[i] != groundtruths[i]:
            misclassified.append(i)
            im1 = cv2.resize(cv2.imread(df['chan2'].iloc[i].replace('chan2', 'chan1')), (224, 224))
            im2 = cv2.resize(cv2.imread(df['chan2'].iloc[i]), (224, 224))
            im3 = cv2.resize(cv2.imread(df['chan3'].iloc[i]), (224, 224))
            im7 = cv2.resize(cv2.imread(df['chan7'].iloc[i]), (224, 224))
            im11 = cv2.resize(cv2.imread(df['chan11'].iloc[i]), (224, 224))

            plot_all_image_channels([im1, im2, im3, im7, im11],
                                    main_title=f"Predicted {class_names[predictions[i]]}, "
                                               f"GroundTruth {class_names[groundtruths[i]]}", save=True,
                                    save_name=f'Image_idx{i}.png')

        if len(misclassified) == n:
            break


save_misclassification(preds, gts, n=10, df=test_df, class_names=class_names)

# Save results
res_df = pd.DataFrame([res])
print(res_df)

res_df.to_csv('test_set_results.csv', index=False)
res_df.to_excel('test_set_results.xlsx', index=False)

print('Time taken:', time.time() - st)
