import os
import pandas as pd
import numpy as np

directory = '../prepare_data_source/revised_clusters/multi'

image_list = os.listdir(directory)

chan2, chan3, chan7, chan11 = [], [], [], []
for image in image_list:
    if 'chan2' in image:
        chan2.append(f'{directory}/{image}')
    elif 'chan3' in image:
        chan3.append(f'{directory}/{image}')
    elif 'chan7' in image:
        chan7.append(f'{directory}/{image}')
    elif 'chan11' in image:
        chan11.append(f'{directory}/{image}')

print(len(chan2), len(chan3), len(chan7), len(chan11))

df = pd.DataFrame(columns=['chan2', 'chan3', 'chan7', 'chan11'])

df['chan2'] = pd.Series(chan2)
df['chan3'] = pd.Series(chan3)
df['chan7'] = pd.Series(chan7)
df['chan11'] = pd.Series(chan11)

print(df)

df.to_csv('multi_data_list.csv', index=False)