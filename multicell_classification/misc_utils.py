import random
import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader
import cv2
import os
import json
import datetime
import shutil
import matplotlib.pyplot as plt
from multicell_classification.cnn import ImageDataset
from prepare_data_source.data_loading import DataLoad


def plot_roc(roc_dict, class_names):
    # Plotting receiver operator characteristics
    # class_names = ['rbc', 'wbc', 'wbc platelet']
    for i in range(len(class_names)):
        fpr, tpr, thresh = roc_curve(roc_dict['all_labels'], roc_dict['all_probs'][:, i], pos_label=i)
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


def plot_accuracy(history, title=None):
    plt.plot(history['train'], label='train')
    plt.plot(history['valid'], label='validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    if title:
        plt.title(f'{title} Accuracy curve')
    else:
        plt.title('Accuracy curve')
    plt.legend()
    plt.show()


def set_seed(random_seed):
    random.seed(42)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)


def get_current_time():
    return datetime.datetime.now().strftime("%d%m%Y_%H%M")


def disp_train_dataset(img_dir, lim=20):
    for path, _, filenames in os.walk(img_dir + '/train'):
        for i, file in enumerate(filenames):
            img_path = path.replace("\\", "/") + '/' + file
            img = cv2.imread(img_path)
            print(f'Image shape:{img.shape}')
            plt.imshow(img)
            plt.show()
            if i > lim:
                break


def mean_std_computation(img_df=None, img_dir=None):
    # Computing mean and std of image dataset for normalization
    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])
    base_ds = ImageDataset(dataframe=img_df, mode='train', transforms=tf)
    loader = DataLoader(base_ds, batch_size=1, shuffle=True)

    mean, std, total_images_count = 0, 0, 0
    for images, _ in loader:
        images = images.view(images.size(0), images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += images.size(0)
    mean /= total_images_count
    std /= total_images_count
    return mean.numpy(), std.numpy()


def check_predictions(img_list, predictions, labels, class_names, lim=15, save=False):
    misclassified = []
    for i, img_path in enumerate(img_list):
        if predictions[i] != labels[i]:
            misclassified.append(i)
            fig = plt.figure(figsize=(7, 7))
            img = cv2.imread(img_path)
            plt.imshow(img)
            plt.title(f'Prediction: {class_names[predictions[i]]}, '
                      f'Groundtruth: {class_names[labels[i]]}')
            plt.axis('off')
            plt.show()
            if save:
                os.makedirs('./p1_misclassifications', exist_ok=True)
                fig.savefig(f'./p1_misclassifications/Misclassification idx_{i}.png')
                plt.close(fig)

            if len(misclassified) == lim:
                break


def get_channel1_unlabelled(path, src, make_dir=False, verbose=False):
    data = DataLoad(main_path=f'{src}', dim=(224, 224))
    data.get_channel1()
    channel1 = data.channel1
    if verbose:
        print('Length of channel1:', len(channel1))

    # A list to keep track of images already labelled
    already_labelled = []

    if os.path.isfile(f'../prepare_data_source/categories.json') and os.access('../prepare_data_source/categories.json',
                                                                               os.R_OK):
        category = json.load(open('../prepare_data_source/categories.json', 'r'))
        for keys in category.keys():
            already_labelled.append(keys)
    if verbose:
        print('Number of images already labelled: ', len(already_labelled))

    for x in already_labelled:
        channel1.remove(x)

    # Now left with the list of channel1 images that haven't been labelled to be used for generating ground-truths
    if make_dir and path:
        if os.path.exists(f'{path}'):
            shutil.rmtree(f'{path}')
        os.makedirs(f'{path}')

        for image_name in channel1:
            shutil.copy(image_name, f'{path}')
        if verbose:
            print(f'Channel1 images to be used for second phase stored in {path}')

    return channel1
