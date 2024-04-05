import matplotlib.pyplot as plt
import cv2
from data_loading import DataLoad
import json
import random
import os
import shutil

data = DataLoad(main_path='MasterDataset/', dim=(224, 224))
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

# Removing already labelled images from channel1 list
for x in already_labelled:
    channel1.remove(x)

print('Length of channel1 after removing labelled images:', len(channel1))

random.shuffle(channel1)

plt.ion()

for image in channel1:
    proceed = input('To label enter any character, press \'n\' to stop:')
    if proceed == 'n':
        break
    print('image filename: ', image)
    plt.imshow(cv2.imread(image))
    plt.pause(0.05)
    print('0. Particles/Unclear, 1. Single Cells, 2. Multiple Cells')
    print('Enter \'p\' if you would like to move to the next image:')
    inp = input('Enter category or \'p\':')
    if inp == 'p':
        print('Image skipped!')
        continue
    else:
        category.update({image: inp})

data.set_labels(category)
data.get_dataframe()

# Saving the class labels that were just annotated
json.dump(category, open("categories.json", 'w'))
print('categories.json has been updated')

