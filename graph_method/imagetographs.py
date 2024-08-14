"""
The first step is to take our images and obtain color vectors, either by color histograms
or by mean color values. This should ideally simplify and reduce dataset size when
building the graph structure.
"""
import os.path as osp
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from color_classify.histogram import get_feature_vector

st = time.time()


# image_csv = "C:/DATA/UGASem5/FeatureExtraction/CNNFeatureExtraction/channelsdata.csv"
#
# df = pd.read_csv(image_csv)
#
# for idx, row in df.iterrows():
#     if osp.isfile(row['chan2']) and osp.isfile(row['chan3']) and osp.isfile(row['chan7']) and osp.isfile(row['chan11']):
#         pass
#     else:
#         df.drop(idx, inplace=True)
#
# df.to_csv('filename_list.csv', index=False)

image_csv = 'filename_list.csv'
df = pd.read_csv(image_csv)
fv = get_feature_vector(df)

# Obtain cosine similarity
similarity_matrix = cosine_similarity(fv)
print(similarity_matrix)

# Construct graph
knn_graph = kneighbors_graph(fv, n_neighbors=3, metric='cosine')
graph = nx.from_scipy_sparse_array(knn_graph)

nx.draw(graph)
plt.savefig("knngraph.png")

print('Time taken:', time.time() - st)

