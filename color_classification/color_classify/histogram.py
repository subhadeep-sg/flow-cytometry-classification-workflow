import cv2
import pandas as pd
import numpy as np
import os


def get_color_histogram(column, channel):
    """
    A function to convert all images based on filenames and channel (RGB) to get color histogram value as output.
    :param column: single dataframe column
    :param channel: 0 for blue, 1 for green, 2 for red
    :return: single dataframe column with color histogram
    """
    new_df = column.copy()
    new_column = new_df.apply(lambda x: cv2.calcHist(cv2.imread(x), channels=[channel],
                                                     mask=None, histSize=[256],
                                                     ranges=[0, 256]))
    return new_column.apply(lambda main_list: [x for sublist in main_list for x in sublist])


def get_average_list(row_list):
    list1, list2, list3 = row_list
    return (np.array(list1) + np.array(list2) + np.array(list3)) / 3


def get_average_histogram(dataframe, data_channel_name):
    ndf = pd.DataFrame()
    # Blue Channel
    ndf['0'] = get_color_histogram(dataframe[data_channel_name], channel=0)
    # Green Channel
    ndf['1'] = get_color_histogram(dataframe[data_channel_name], channel=1)
    # Red Channel
    ndf['2'] = get_color_histogram(dataframe[data_channel_name], channel=2)

    return ndf[['0', '1', '2']].apply(lambda x: get_average_list(x), axis=1)


def get_feature_vector(dataframe):
    df2 = pd.DataFrame()
    df2['chan2'] = get_average_histogram(dataframe, 'chan2')
    df2['chan3'] = get_average_histogram(dataframe, 'chan3')
    df2['chan7'] = get_average_histogram(dataframe, 'chan7')
    df2['chan11'] = get_average_histogram(dataframe, 'chan11')

    # Obtaining feature vector
    feature_vector = []
    for index, row in df2.iterrows():
        # feature_vector.append(np.stack((row['chan2'], row['chan3'], row['chan7'], row['chan11']), axis=-1))
        feature_vector.append(np.concatenate((row['chan2'], row['chan3'], row['chan7'], row['chan11'])))

    del df2
    print("Feature vector shape for all images: ", np.array(feature_vector).shape)
    return np.array(feature_vector)
