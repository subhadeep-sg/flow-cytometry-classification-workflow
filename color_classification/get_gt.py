import numpy as np
import pandas as pd
import json
from PIL import Image
import time
import cv2

"""
Color criteria:
Presence of green implies platelets
purple
red


"""
st = time.time()
# df = pd.read_csv('../graph_method/filename_list.csv', index_col=False)
# df = df.drop(columns=['Unnamed: 0'])
# df = pd.read_csv('../prepare_data_source/filename_list.csv', index_col=False)
df = pd.read_csv('./multi_data_list.csv', index_col=False)
chan2_list = df['chan2'].tolist()
chan3_list = df['chan3'].tolist()
chan7_list = df['chan7'].tolist()
chan11_list = df['chan11'].tolist()

chan_set = set(chan2_list)
assert len(chan_set) == len(chan2_list), "Check for duplicate image files"


def search_colors(img):
    color_ranges = {
        'green': [(36, 50, 70), (89, 255, 255)],
        'yellow': [(20, 50, 70), (35, 255, 255)],
        'purple': [(129, 50, 70), (158, 255, 255)],
        'red1': [(0, 50, 70), (9, 255, 255)],
        'red2': [(159, 50, 70), (180, 255, 255)],  # Red can span two ranges
    }
    image = cv2.imread(img)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    colors_found = {color: False for color in color_ranges}

    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))

        if np.any(mask):  # If the mask has any non-zero values, the color exists
            colors_found[color] = True

    red_mask1 = cv2.inRange(hsv_image, np.array(color_ranges['red1'][0]), np.array(color_ranges['red1'][1]))
    red_mask2 = cv2.inRange(hsv_image, np.array(color_ranges['red2'][0]), np.array(color_ranges['red2'][1]))
    if np.any(red_mask1) or np.any(red_mask2):
        colors_found['red'] = True
    else:
        colors_found['red'] = False

    # print("Colors found:", colors_found)
    return colors_found


df['label'] = df['chan2'].copy()
wbc, wbc_platelet, rbc, platelet = [], [], [], []
for i, img in enumerate(chan2_list):
    colors = search_colors(img)
    colors3 = search_colors(chan3_list[i])
    colors7 = search_colors(chan7_list[i])
    colors11 = search_colors(chan11_list[i])
    """
    Green(c2) + No (Yellow(c3) or Purple(c7) or Red(c11)) = Platelet
    Green(c2) + (Yellow(c3) or Purple(c7) or Red(c11)) = WBC_Platelet
    No Green(c2) + (Yellow(c3) or Purple(c7) or Red(c11)) = WBC
    None = RBC
    """
    if colors['green'] is True and (colors3['yellow'] or colors7['yellow'] or colors11['red']) is True:
        wbc_platelet.append(i)
        df.at[i, 'label'] = 'wbc platelet'
    elif colors3['yellow'] or colors7['purple'] or colors11['red'] is True:
        wbc.append(i)
        df.at[i, 'label'] = 'wbc'
    elif colors['green']:
        platelet.append(i)
        df.at[i, 'label'] = 'platelet'
    else:
        rbc.append(i)
        df.at[i, 'label'] = 'rbc'

print(f'Length of the four cluster types: {len(wbc)}, {len(wbc_platelet)}, {len(rbc)}, {len(platelet)}')

print(df)
df.to_csv('../color_classification/groundtruths.csv', index=False)

print('Time taken: ', time.time() - st)
