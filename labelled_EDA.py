import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

df = pd.read_csv('small.csv')

category = json.load(open('categories.json'))

print('Length of dictionary:', len(category))

print(df.class_label.value_counts())
print('Total images in dataframe:', len(df.class_label))

print('class_weights = ')
count_0 = df.class_label.value_counts()[0]
count_1 = df.class_label.value_counts()[1]
count_2 = df.class_label.value_counts()[2]
total = count_2 + count_1 + count_0
print(2, count_2 / total)
print(1, count_1 / total)
print(0, count_0 / total)

plt.bar(['Particles', 'Single', 'Multi'], [count_0, count_1, count_2], )
plt.xlabel("Cluster type")
plt.ylabel("Image count for each class")
plt.title("Class Distribution")
plt.show()
