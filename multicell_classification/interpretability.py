import torch
import cv2
import numpy as np
import pandas as pd
import time
import os
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from misc_utils import set_seed
from multicell_classification.cnn import ImageDataset
from multicell_classification.binary_test import test_model

st = time.time()
set_seed(42)
img_directory = './classify_dataset/test'


# def test_model(cnn, loader, loss_fn, class_weights,
#                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
#     test_correct, test_total = 0, 0
#     test_running_loss = 0.0
#     class_correct, class_total = [0, 0, 0], [0, 0, 0]
#     all_pred, all_labels, all_probs = [], [], []
#     results = {'accuracy': None,
#                'f1': None,
#                'precision': None,
#                'recall': None,
#                'roc-auc': None,
#                'class_weights': class_weights,
#                'loss': None,
#                'all_pred': None,
#                'all_labels': None,
#                'all_probs': None
#                }
#     with torch.no_grad():
#         for inputs, labels in loader:
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             outputs = cnn(inputs)
#             prob = F.softmax(outputs, dim=1)
#             _, pred = torch.max(outputs, 1)
#             test_correct += (pred == labels).sum().item()
#             test_total += labels.size(0)
#
#             # Weighted Accuracy
#             for i in range(3):
#                 class_idx = (labels == i)
#                 class_correct[i] += (pred[class_idx] == i).sum().item()
#                 class_total[i] += class_idx.sum().item()
#
#             all_pred.extend(pred.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
#             all_probs.append(prob.cpu().numpy())
#
#             loss = loss_fn(outputs, labels)
#             test_running_loss += loss.item()
#
#     all_probs = np.concatenate(all_probs, axis=0)
#
#     class_accuracies = [correct / total if total > 0 else 0 for correct, total in zip(class_correct, class_total)]
#     weighted_accuracy = sum(weight * ca for weight, ca in zip(class_weights, class_accuracies)) / class_weights.sum()
#     f1_2 = f1_score(all_labels, all_pred, labels=[1], average='macro')
#     f1_1 = f1_score(all_labels, all_pred, labels=[0], average='macro')
#
#     results['accuracy'] = round(weighted_accuracy.item() * 100, 4)
#     results['f1'] = round(f1_score(all_labels, all_pred, average='macro'), 4)
#     results['roc-auc'] = round(roc_auc_score(all_labels, all_probs[:, 1], multi_class='ovo'), 4)
#     results['precision'] = round(precision_score(all_labels, all_pred, average='macro'), 4)
#     results['recall'] = round(recall_score(all_labels, all_pred, average='macro'), 4)
#     results['all_pred'] = all_pred
#     results['all_labels'] = all_labels
#     results['all_probs'] = all_probs
#     results['loss'] = round(test_running_loss, 4)
#
#     conf_matrix = confusion_matrix(all_labels, all_pred, labels=[0, 1])
#     plt.figure()
#     disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Cluster', 'Non Cluster'])
#     disp.plot()
#     plt.tight_layout()
#     plt.show()
#
#     test_acc = (test_correct / test_total) * 100
#     print(f'Test accuracy:{test_acc:.4f}')
#     print(
#         f"Weighted Accuracy: {weighted_accuracy * 100:.4f}, F1 Score('Non Cluster'): {f1_2:.4f}, F1 Score('Cluster'): {f1_1:.4f}")
#     print(f"roc-auc: {results['roc-auc']:.4f}, precision: {results['precision']:.4f}, recall: {results['recall']:.4f}")
#     print(f"F1 Score ('Average'): {results['f1']:.4f}, loss: {results['loss']:.4f}")
#
#     return results
def run_gradcam_corrects(model, test_ds, preds, labels, model_name='SimpleConvNet', target_layer='conv3', num_images=5,
                         device='cuda'):
    model.to(device)
    model.eval()
    cam_extractor = GradCAM(model, target_layer=target_layer)

    correct_indices = [i for i in range(len(preds)) if preds[i] == labels[i]]
    selected_indices = correct_indices[:num_images]

    label_names = ['cluster', 'non cluster']
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.6558058, 0.6558058, 0.6558058), (0.11742277, 0.11742277, 0.11742277))
    ])

    original_images, overlays, titles = [], [], []

    for idx in selected_indices:
        img_path = test_ds.get_image_list()[idx]
        orig = cv2.imread(img_path)
        orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(orig_rgb).resize((224, 224))

        img_tensor = transform(orig_rgb).unsqueeze(0).to(device)
        output = model(img_tensor)
        pred_class = output.argmax(dim=1).item()
        cam = cam_extractor(pred_class, output)[0].squeeze().cpu().numpy()
        cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(pil_img.size)

        overlay = overlay_mask(pil_img, cam_img, alpha=0.5)

        original_images.append(pil_img)
        overlays.append(overlay)
        titles.append(f"{model_name}\nGT & Pred: {label_names[labels[idx]]}")

    # Plot original images
    fig1, axes1 = plt.subplots(1, len(original_images), figsize=(18, 5))
    for ax, img, title in zip(axes1.ravel(), original_images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    fig1.suptitle("Correctly Classified - Original Images", fontsize=16)
    plt.tight_layout()
    plt.show()

    # Plot overlays
    fig2, axes2 = plt.subplots(1, len(overlays), figsize=(18, 5))
    for ax, img, title in zip(axes2.ravel(), overlays, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    fig2.suptitle("Correctly Classified - Grad-CAM", fontsize=16)
    plt.tight_layout()
    plt.show()


def save_misclassified_images(model_name, test_ds, preds, labels, output_dir='appendix_misclass', max_images=None):
    os.makedirs(os.path.join(output_dir, model_name), exist_ok=True)
    label_names = ['cluster', 'non_cluster']
    misclassified_indices = [i for i in range(len(preds)) if preds[i] != labels[i]]

    if max_images:
        misclassified_indices = misclassified_indices[:max_images]

    for idx in misclassified_indices:
        img_path = test_ds.get_image_list()[idx]
        orig = cv2.imread(img_path)
        orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

        # Annotate the image with GT and predicted
        title = f"GT: {label_names[labels[idx]]}, Pred: {label_names[preds[idx]]}"
        annotated_img = cv2.putText(orig_rgb.copy(), title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (255, 0, 0), 2, cv2.LINE_AA)

        # Save image
        base_filename = os.path.basename(img_path)
        save_path = os.path.join(output_dir, model_name, f"{idx}_{base_filename}")
        cv2.imwrite(save_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))


def run_gradcam(model, test_ds, preds, labels, model_name='SimpleConvNet', target_layer='conv3', num_images=5,
                device='cuda'):
    """
    Visualize Grad-CAM on misclassified test examples.
    Shows two figures: one with original images, one with Grad-CAM overlays.
    """

    from torchvision.transforms.functional import to_pil_image

    model.to(device)
    model.eval()
    cam_extractor = GradCAM(model, target_layer=target_layer)

    # Find misclassified indices
    misclassified_indices = [i for i in range(len(preds)) if preds[i] != labels[i]]
    selected_indices = misclassified_indices[:num_images]

    label_names = ['cluster', 'non cluster']
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.6558058, 0.6558058, 0.6558058), (0.11742277, 0.11742277, 0.11742277))
    ])

    original_images = []
    overlays = []
    titles = []

    for idx in selected_indices:
        img_path = test_ds.get_image_list()[idx]
        orig = cv2.imread(img_path)
        orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(orig_rgb).resize((224, 224))

        img_tensor = transform(orig_rgb).unsqueeze(0).to(device)

        # Forward pass and get CAM
        output = model(img_tensor)
        pred_class = output.argmax(dim=1).item()
        cam = cam_extractor(pred_class, output)[0].squeeze().cpu().numpy()
        cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(pil_img.size)

        overlay = overlay_mask(pil_img, cam_img, alpha=0.5)

        original_images.append(pil_img)
        overlays.append(overlay)
        titles.append(f"{model_name}\nGT: {label_names[labels[idx]]}, Pred: {label_names[preds[idx]]}")

    # Show original images
    fig1, axes1 = plt.subplots(1, len(original_images), figsize=(18, 5))
    for ax, img, title in zip(axes1.ravel(), original_images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    fig1.suptitle("Original Images (Misclassified)", fontsize=16)
    plt.tight_layout()
    plt.show()

    # Show Grad-CAM overlays
    fig2, axes2 = plt.subplots(1, len(overlays), figsize=(18, 5))
    for ax, img, title in zip(axes2.ravel(), overlays, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    fig2.suptitle("Grad-CAM", fontsize=16)
    plt.tight_layout()
    plt.show()


# def run_gradcam(model, test_ds, preds, labels, model_name='SimpleConvNet', target_layer='conv3', num_images=5, device='cuda'):
#     """
#     Visualize Grad-CAM on misclassified test examples.
#     :param model: Trained model
#     :param test_ds: Test dataset
#     :param preds: Predictions from test_model()
#     :param labels: Ground-truth labels from test_model()
#     :param target_layer: Layer name to apply Grad-CAM on
#     :param num_images: Number of misclassified examples to visualize
#     :param device: 'cuda' or 'cpu'
#     """
#
#     model.to(device)
#     model.eval()
#     cam_extractor = GradCAM(model, target_layer=target_layer)
#
#     # Find misclassified indices
#     misclassified_indices = [i for i in range(len(preds)) if preds[i] != labels[i]]
#     selected_indices = misclassified_indices[:num_images]
#
#     label_names = ['cluster', 'non cluster']
#     transform = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.6558058, 0.6558058, 0.6558058), (0.11742277, 0.11742277, 0.11742277))
#     ])
#
#     fig, axes = plt.subplots(1, len(selected_indices), figsize=(18, 5))
#     axes = axes.ravel()
#
#     for idx, i in enumerate(selected_indices):
#         img_path = test_ds.get_image_list()[i]
#         orig = cv2.imread(img_path)
#         orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
#         pil_img = Image.fromarray(orig_rgb).resize((224, 224))
#
#         # Preprocess
#         img_tensor = transform(orig_rgb).unsqueeze(0).to(device)
#
#         # Forward pass and get CAM
#         output = model(img_tensor)
#         pred_class = output.argmax(dim=1).item()
#         cam = cam_extractor(pred_class, output)[0].squeeze().cpu().numpy()
#         cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(pil_img.size)
#
#         # Overlay CAM on original
#         overlay = overlay_mask(pil_img, cam_img, alpha=0.5)
#
#         axes[idx].imshow(overlay)
#         axes[idx].set_title(f"{model_name}, GT: {label_names[labels[i]]}\nPred: {label_names[preds[i]]}")
#         axes[idx].axis('off')
#
#     plt.tight_layout()
#     plt.show()


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

class_weights = [0.5, 0.5]
class_names = ['cluster_revised', 'non_cluster_revised']
class_weights = torch.FloatTensor(class_weights).cuda()
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = f'./saved_models/19-01_2146_90_model.pt'
model = torch.load(model_path, weights_only=False)
# print(model)
model.eval()
results = test_model(model, loader=test_dataloader,
                     class_weights=class_weights,
                     loss_fn=loss_fn, device=device, plot=False)

preds = results['all_pred']
labels = results['all_labels']
image_list = test_ds.get_image_list()

# run_gradcam(model, test_ds, preds, labels, target_layer='conv3')

# run_gradcam_corrects(model, test_ds, preds, labels, target_layer='conv3')

# save_misclassified_images(model_name='SimpleCNN', test_ds=test_ds, preds=preds, labels=labels)


pretrained_models = [
    '../pretrained/saved_models/ResNet_06-02_0524_91_model.pt',
    '../pretrained/saved_models/EfficientNet_06-02_0735_93_model.pt',
    '../pretrained/saved_models/MobileNetV2_06-02_0749_92_model.pt',
    '../pretrained/saved_models/DenseNet_06-02_0901_92_model.pt'
]

model_names = ['ResNet', 'EfficientNet', 'MobileNetV2', 'DenseNet']
target_layer_names = ['layer4', 'features.7', 'features.18', 'features.denseblock4']
for i, mod_path in enumerate(pretrained_models):
    model = torch.load(mod_path, weights_only=False)
    model.eval()
    results = test_model(model, loader=test_dataloader,
                         class_weights=class_weights,
                         loss_fn=loss_fn, device=device, plot=False)

    predictions = results['all_pred']
    labels = results['all_labels']
    # run_gradcam(model, test_ds, predictions, labels,
    #             target_layer=target_layer_names[i],
    #             model_name=model_names[i])

    run_gradcam_corrects(model, test_ds, predictions, labels,
                         target_layer=target_layer_names[i],
                         model_name=model_names[i])

    # save_misclassified_images(model_name=model_names[i], test_ds=test_ds, preds=predictions, labels=labels)

print('Runtime', time.time() - st)
