import os
import numpy as np
import pandas as pd
import shutil

# To create a specific batch size folder
# I'll need to make a pre-existing csv folder where I can store filenames and conveniently place my predictions to merge.

test_path = '../multicell_classification/classify_dataset/test'

test_df = pd.DataFrame(columns=['filepath', 'filename', 'label', 'assigned_batch'])

filepaths, filenames, labels = [], [], []

for dir_path, dirs, files in os.walk(test_path):
    for filename in files:
        main_dir = dir_path.replace('\\', '/')
        filepaths.append(f"{main_dir}/{filename}")
        filenames.append(filename)
        labels.append(dir_path.split('\\')[-1])

labels = ['cluster' if x == 'cluster_revised' else 'non cluster' for x in labels]

test_df['filepath'] = filepaths
test_df['filename'] = filenames
test_df['label'] = labels

shuffled_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
print(shuffled_df)


def create_batched_directory(dataframe, per_batch=5, image_num=314):
    total_batches = round(image_num / per_batch)
    print(f'Total Batches: {total_batches}')
    os.makedirs('test_batched', exist_ok=True)

    assigned = [i for i in range(total_batches) for _ in range(per_batch)]
    dataframe['assigned_batch'] = assigned

    for i in range(len(dataframe)):
        os.makedirs(f"test_batched/{dataframe['assigned_batch'][i]}", exist_ok=True)
        shutil.copyfile(dataframe['filepath'][i], f"test_batched/{dataframe['assigned_batch'][i]}/{dataframe['filename'][i]}")

    return dataframe


test_df = create_batched_directory(dataframe=shuffled_df, per_batch=5, image_num=len(test_df))
test_df.to_csv('test_batched.csv', index=False)
print(test_df)
