import os

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Paths
# chan1_folder = '../multicell_classification/classify_dataset/train'  # e.g., multi, single, particles
chan1_folder = '../dataset/prepared_revised/train'
master_dataset = '../MasterDataset'
mask_output_dir = 'chan1_masks'
filtered_chan1 = 'filtered_chan1'
os.makedirs(mask_output_dir, exist_ok=True)
os.makedirs(filtered_chan1, exist_ok=True)


def get_all_images(root_dir, extensions='.png'):
    image_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith(extensions) and '_chan1_' in f:
                image_paths.append(f)
    return image_paths


chan1_files = get_all_images(chan1_folder)

print(chan1_files)

for f in chan1_files:
    chan2_name = f.replace('_chan1_', '_chan2_')
    chan3_name = f.replace('_chan1_', '_chan3_')
    chan7_name = f.replace('_chan1_', '_chan7_')
    chan11_name = f.replace('_chan1_', '_chan11_')

    merged = np.zeros((224, 224, 3), dtype=np.uint8)  # RGB image initialized to zero

    if os.path.exists(master_dataset + '/' + chan2_name):
        chan2 = cv2.imread(master_dataset + '/' + chan2_name, cv2.IMREAD_GRAYSCALE)
        chan2 = cv2.resize(chan2, (224, 224))
        merged[..., 1] = chan2  # Green channel

    if os.path.exists(master_dataset + '/' + chan3_name):
        chan3 = cv2.imread(master_dataset + '/' + chan3_name, cv2.IMREAD_GRAYSCALE)
        chan3 = cv2.resize(chan3, (224, 224))
        merged[..., 2] = cv2.add(merged[..., 2], chan3)  # Add to red

    if os.path.exists(master_dataset + '/' + chan7_name):
        chan7 = cv2.imread(master_dataset + '/' + chan7_name, cv2.IMREAD_GRAYSCALE)
        chan7 = cv2.resize(chan7, (224, 224))
        merged[..., 0] = chan7  # Blue channel

    if os.path.exists(master_dataset + '/' + chan11_name):
        chan11 = cv2.imread(master_dataset + '/' + chan11_name, cv2.IMREAD_GRAYSCALE)
        chan11 = cv2.resize(chan11, (224, 224))
        merged[..., 2] = cv2.add(merged[..., 2], chan11)  # Add to red

    gray = cv2.cvtColor(merged, cv2.COLOR_BGR2GRAY)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # enhanced = clahe.apply(gray)
    # _, gray = cv2.threshold(enhanced, 10, 255, cv2.THRESH_BINARY)


    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)


    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)


    overlay = merged.copy()
    overlay[mask > 0] = [0, 0, 255]  # red overlay on masked region

    # plt.figure(figsize=(12, 5))
    #
    # plt.subplot(1, 2, 1)
    # plt.title("Binary Mask")
    # plt.imshow(mask, cmap='gray')
    # plt.axis('off')
    #
    # plt.subplot(1, 2, 2)
    # plt.title("Mask Overlay on Original")
    # plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    #
    # plt.tight_layout()
    # plt.show()


    mask_name = f.replace('_chan1_', '_mask_')  # or just replace('.png', '_mask.png')
    # mask_name = mask_name.replace('.png', '')
    cv2.imwrite(os.path.join(mask_output_dir, mask_name), mask)

for filename in chan1_files:
    img = cv2.imread(master_dataset + '/' + filename)
    img = cv2.resize(img, (224, 224))
    mask = cv2.imread(mask_output_dir + '/' + filename.replace('_chan1_', '_mask_'), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (224, 224))


    masked_img = cv2.bitwise_and(img, img, mask=mask)
    cv2.imwrite(os.path.join(filtered_chan1, filename), masked_img)


    # plt.imshow(masked_img)
    # plt.show()


