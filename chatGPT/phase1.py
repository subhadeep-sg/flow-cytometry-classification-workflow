import os
import numpy as np
import pandas as pd
import cv2
import base64
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

clust_source = '../multicell_classification/classify_dataset/train/cluster_revised'
non_clust_source = '../multicell_classification/classify_dataset/train/non_cluster_revised'


#

# predictions = valid_predictions['Predicted Label']
# image_names = valid_predictions['Image']
# gt = []
# for image in image_names:
#     if image in os.listdir(clust_source):
#         gt.append('cluster')
#     else:
#         gt.append('non cluster')
#
# print(classification_report(gt, predictions))


training_csv = pd.read_csv('training_data_export.csv')
images = training_csv['Filename'].tolist()

for img in images:
    filename = f'fewshotdata/train'
    im = cv2.imread(img)
    plt.imshow(im)
    plt.show()
    break
