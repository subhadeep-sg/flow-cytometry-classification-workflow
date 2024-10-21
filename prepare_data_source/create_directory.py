import numpy as np
import pandas as pd
import os
import shutil
import splitfolders
import time
"""
Script to create dataset folder. Will delete any existing folders and replace with new ones.
"""

st = time.time()
# Create new directory for dataset based on manual labels
# if os.path.exists('../dataset'):
#     shutil.rmtree('../dataset')

if os.path.exists('../dataset/prepared_revised'):
    shutil.rmtree('../dataset/prepared_revised')
    os.makedirs('../dataset/prepared_revised', exist_ok=False)

# if os.path.exists('../prepared_dataset'):
#     shutil.rmtree('../prepared_dataset')

# if os.path.exists('../dataset/prepared_dataset'):
#     shutil.rmtree('../dataset/prepared_dataset')
# print('Pre-existing directories deleted.')
# #
# os.makedirs('../dataset')
# os.makedirs('../dataset/raw')
# os.makedirs('../dataset/raw/particles')
# os.makedirs('../dataset/raw/single')
# os.makedirs('../dataset/raw/multi')
#
# df = pd.read_csv('../small.csv')
# filenames = df.image_name.to_list()
# label = df.class_label.to_list()
#
# iter = 0
# for image_name in filenames:
#     if label[iter] == 2:
#         shutil.copy(image_name, '../dataset/raw/multi')
#     elif label[iter] == 1:
#         shutil.copy(image_name, '../dataset/raw/single')
#     elif label[iter] == 0:
#         shutil.copy(image_name, '../dataset/raw/particles')
#     else:
#         print('Something went wrong!')
#     iter += 1
# print('Files sorted into multi, single and particles in dataset directory')



# Adding a block to split the dataset into train validation and test
splitfolders.ratio("../dataset/raw_revised_HL/raw",  # The location of dataset
                   output="../dataset/prepared_revised",  # The output location
                   seed=42,  # The number of seed
                   ratio=(.6, .2, .2),  # The ratio of split dataset
                   group_prefix=None,  # If your dataset contains more than one file like ".jpg", ".pdf", etc
                   move=False  # If you choose to move, turn this into True
                   )
print('Dataset directory further organized into train, test, validation in prepared_directory')
print('Data directory creation complete')
print('Time taken: ', time.time()-st)
