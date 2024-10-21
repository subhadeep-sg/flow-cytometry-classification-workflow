import numpy as np
import pandas as pd
import os


df = pd.read_csv('clust_non_clust.csv', index_col=False)

# Add other channels
chan1 = df['filename'].tolist()
print(len(chan1))

revised_chan2 = []
list3 = []
list7 = []
list11 = []
for img in chan1:
    img = img.replace('/dataset/raw_revised_HL/raw/multi', '/MasterDataset')
    chan2 = img.replace('chan1', 'chan2')
    chan3 = img.replace('chan1', 'chan3')
    chan7 = img.replace('chan1', 'chan7')
    chan11 = img.replace('chan1', 'chan11')

    if os.path.exists(chan2) and os.path.exists(chan3) and os.path.exists(chan7) and os.path.exists(chan11):
        revised_chan2.append(chan2)
        list3.append(chan3)
        list7.append(chan7)
        list11.append(chan11)

new_df = df.copy()
new_df['chan2'] = pd.Series(revised_chan2)
new_df['chan3'] = pd.Series(list3)
new_df['chan7'] = pd.Series(list7)
new_df['chan11'] = pd.Series(list11)

new_df = new_df.drop(columns=['filename', 'class_label'])
new_df = new_df.dropna(axis=0, how='any')

new_df.to_csv('filename_list.csv', index=False)

print(new_df)





