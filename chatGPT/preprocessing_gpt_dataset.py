import numpy as np
import pandas as pd
import os
import cv2
import time
import matplotlib.pyplot as plt

st = time.time()
clust_train = './train_set'
clust_test = './test_batched'

trains = {}
for root, subdir, filename in os.walk(clust_train):
    for file in filename:
        if file.endswith('.png'):
            # trains.append(root.replace('\\', '/') + '/' + file)
            fpath = root.replace('\\', '/') + '/' + file
            label = root.split('\\')[-2]
            # print(label)
            trains[fpath] = label
print(len(trains))

test_dict = {}
for root, subdir, filename in os.walk(clust_test):
    for file in filename:
        if file.endswith('.png'):
            # trains.append(root.replace('\\', '/') + '/' + file)
            fpath = root.replace('\\', '/') + '/' + file
            # label = root.split('\\')[-2]
            batch_name = root.split('\\')[-1]
            # print(root)
            # print(batch_name)
            test_dict[fpath] = batch_name
print(len(test_dict))
# print(test_dict)


def preprocess_pipeline(raw_img):
    raw_img = cv2.imread(raw_img, cv2.IMREAD_GRAYSCALE)
    raw_img = cv2.GaussianBlur(raw_img, (5, 5), 5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_raw_img = clahe.apply(raw_img)
    _, thresh_raw_image = cv2.threshold(enhanced_raw_img, 50, 255,
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh_raw_image


# os.makedirs('./processed_train_set', exist_ok=True)
# os.makedirs('./processed_train_set/Cluster', exist_ok=True)
# os.makedirs('./processed_train_set/Non cluster', exist_ok=True)
#
# for i, image in enumerate(trains.keys()):
#     print(image)
#     img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
#     thresh = preprocess_pipeline(image)
#
#     if 5 < i < 10:
#         f, ax = plt.subplots(1, 2)
#         ax[0].imshow(img)
#         ax[1].imshow(thresh)
#         plt.show()
#
#     cv2.imwrite(f'./processed_train_set/{trains[image]}/image{i}.png', thresh)

for i, image in enumerate(test_dict.keys()):
    # print(image)
    fname = image.split('/')[-1]
    print(fname)
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    thresh = preprocess_pipeline(image)

    if 5 < i < 10:
        f, ax = plt.subplots(1, 2)
        ax[0].imshow(img)
        ax[1].imshow(thresh)
        plt.show()

    os.makedirs('./processed_test_set', exist_ok=True)
    os.makedirs(f'./processed_test_set/{test_dict[image]}', exist_ok=True)
    cv2.imwrite(f'./processed_test_set/{test_dict[image]}/{fname}.png', thresh)


print('Run time: ', time.time() - st)
