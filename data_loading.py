import numpy as np
import pandas as pd
import os
import cv2
import json

"""
This file defines class DataLoad to read the MasterDataset and generate a dataframe
with the images and labels for training.
"""


class DataLoad:
    """
    Class to store/generate input images based on directory


    """

    def __init__(self, main_path=None, dim=(224, 224), image_list=[]):
        # Height and width of the image for uniformity
        self.h = dim[0]
        self.w = dim[1]

        self.main_path = main_path

        # List storing all images from channel 1
        self.channel1 = image_list

        self.DataFrame = {
            'image_name': [],
            'image_array': [],
            'class_label': []
        }
        self.manual_labels = None

    def get_channel1(self):
        """
        Takes the directory of the images, selects the channel1 images,
        and stores their filenames in a list.
        :return: None
        """
        for file in os.listdir(self.main_path):
            if '_chan1_' in file:
                self.DataFrame['image_name'].append(file)
                self.channel1.append(self.main_path + file)
        if len(self.channel1) > 0:
            print('Added all channel1 images successfully')
        else:
            print('No channel1 images added')

    def get_config(self, labels_path):
        """
        Accepts the .json file containing labels of any number of images and stores
        them in a dictionary.
        :param labels_path: Directory of .json file
        :return: None
        """
        self.manual_labels = json.load(open(labels_path))

    def set_labels(self, labels):
        self.manual_labels = labels

    def get_labels(self):
        return self.manual_labels

    def get_dataframe(self):
        df = {'class_label': [], 'image_array': [], 'image_name': []}
        for i in self.manual_labels:
            df['image_name'].append(i)
            df['class_label'].append(self.manual_labels[i])
            img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.h, self.w))
            img = np.expand_dims(img, 2)  # .flatten()
            df['image_array'].append(img)

        pd.DataFrame(df).to_csv(path_or_buf='small.csv', float_format='%.0f')
        print('DataFrame stored in \'small.csv\'')

        return pd.DataFrame(df)
