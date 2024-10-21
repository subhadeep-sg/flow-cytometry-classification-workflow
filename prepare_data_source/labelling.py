import matplotlib.pyplot as plt
import cv2
from data_loading import DataLoad
import json
import random
import os
import shutil

"""
Purpose of file is to create the training dataset by manually labelling
the multi-cluster images and creating the not needed but to be filtered out
classes of single and particle images (as well as multiple cell images that are NOT clusters)
"""

data = DataLoad(main_path='../MasterDataset/', dim=(224, 224))
data.get_channel1()

channel1 = data.channel1
print('Length of channel1:', len(channel1))

# A list to keep track of images already labelled
already_labelled = []

# Initializing the dictionary to store images and labels
category = {}

if os.path.isfile('categories.json') and os.access('categories.json', os.R_OK):
    category = json.load(open('categories.json', 'r'))
    for keys in category.keys():
        already_labelled.append(keys)
print('Number of images already labelled: ', len(already_labelled))

# Removing already labelled images from channel1 list
if len(already_labelled) == 0:
    pass
else:
    for x in already_labelled:
        channel1.remove(x)
    print('Removing labelled images from Channel 1 list...')
    print('Length of channel1 after removing:', len(channel1))

random.shuffle(channel1)

plt.ion()

labelled_flag = 0
for image in channel1:
    proceed = input('To label enter any character else press \'n\' to stop:')
    if labelled_flag>0:
        print(f'Labelled {labelled_flag} images this session.')
    if proceed == 'n':
        break
    print('image filename: ', image)
    plt.imshow(cv2.imread(image))
    plt.pause(0.05)
    print('0. Particles/Unclear, 1. Single Cells, 2. Multiple Cells')
    print('Enter \'p\' if you would like skip current image:')
    inp = input('Enter category or \'p\':')
    if inp == 'p':
        print('Image skipped!')
        continue
    else:
        category.update({image: inp})
        labelled_flag += 1

data.set_labels(category)
data.get_dataframe()

# Saving the class labels that were just annotated
json.dump(category, open("categories.json", 'w'))
print('categories.json has been updated')
