import pandas as pd
import numpy as np
import os

raw_root = '../dataset/raw_revised_HL/raw'

image_dict = []
labels = []
for dirpath, dirname, filename in os.walk(raw_root):
    for file in filename:
        image_path = dirpath.replace('\\','/')+'/'+file
        image_dict.append(image_path)
        labels.append(dirpath.split('\\')[-1])
        # image_dict['filename'].append(image_path)
        # image_dict['label'].append(dirpath.split('\\')[-1])

df = pd.DataFrame(columns=['filename', 'class_label'])
df['filename'] = pd.Series(image_dict)
df['class_label'] = pd.Series(labels)
print(df)

df.to_csv('clust_non_clust.csv', index=False)

