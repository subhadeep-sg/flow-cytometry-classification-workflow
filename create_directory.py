import numpy as np
import pandas as pd
import os
import shutil
import splitfolders

# Create new directory for dataset based on manual labels
if os.path.exists('dataset'):
    shutil.rmtree('dataset')

os.makedirs('dataset')
os.makedirs('dataset/particles')
os.makedirs('dataset/single')
os.makedirs('dataset/multi')

df = pd.read_csv('small.csv')
filenames = df.image_name.to_list()
label = df.class_label.to_list()

iter = 0
for image_name in filenames:
    if label[iter] == 2:
        shutil.copy(image_name, 'dataset/multi')
    elif label[iter] == 1:
        shutil.copy(image_name, 'dataset/single')
    elif label[iter] == 0:
        shutil.copy(image_name, 'dataset/particles')
    else:
        print('Something went wrong!')
    iter += 1

# Adding a block to split the dataset into train validation and test
splitfolders.ratio("dataset",  # The location of dataset
                   output="prepared_dataset",  # The output location
                   seed=42,  # The number of seed
                   ratio=(.6, .2, .2),  # The ratio of split dataset
                   group_prefix=None,  # If your dataset contains more than one file like ".jpg", ".pdf", etc
                   move=False  # If you choose to move, turn this into True
                   )

print('Data directory creation complete')
