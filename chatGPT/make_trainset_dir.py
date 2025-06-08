import os
import shutil
import pandas as pd

directory = 'train_set'

src = '../multicell_classification/classify_dataset/train'

if not os.path.exists(directory):
    os.makedirs(directory)

df = pd.read_csv('session3ver2.csv', index_col=False)

filenames = df['filename'].tolist()
df['src path'] = df['filename'].copy()

for full_path, _, files in os.walk(src):
    for filename in files:
        subdir = full_path.split('\\')[-1]
        # print(subdir)
        # break
        if filename in filenames and filename.endswith('.png'):
            src_path = src+'/'+subdir+'/'+filename
            idx = filenames.index(filename)
            # print(idx)
            # break
            df.loc[idx, "src path"] = src_path

source_paths_list = df['src path'].tolist()

for i, images in enumerate(source_paths_list):
    shutil.copyfile(images, directory+'/'+filenames[i])


