import json
import numpy as np


def saving_feature_vector(feat_vec):
    with open('featurevector.txt', 'w') as f:
        json.dump(feat_vec.tolist(), f)


def reading_feature_vector(file, verbose=True, as_arr=False, dtype='float64'):
    with open(file) as f:
        arr = json.load(f)
        if verbose:
            print("Importing feature vector of type: ", type(arr))

        if as_arr and dtype:
            print('Converted to type: {}'.format(type(np.array(arr))))
            print('Shape of feature vector: {}'.format(np.array(arr).shape))
            return np.array(arr).astype(dtype)
        else:
            print('Length of list: {}'.format(len(arr)))
            return arr
