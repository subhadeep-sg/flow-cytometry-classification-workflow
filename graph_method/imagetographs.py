"""
The first step is to take our images and obtain color vectors, either by color histograms
or by mean color values. This should ideally simplify and reduce dataset size when
building the graph structure.
"""
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from color_classification.color_classify.histogram import get_feature_vector
from color_classification.color_classify.feature_vectors import saving_feature_vector

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

example_vector = fv[1]

px = 256
example_hist = example_vector[px:px*2]
colors = ('red', 'green', 'blue')
plt.figure(figsize=(7, 2))
plt.plot(example_hist, color='green')
plt.fill_between(range(256), example_hist.flatten(), color='green', alpha=0.3)
plt.title('Average Color Histogram')
plt.show()

def plot_images_stack(df):


# Obtain cosine similarity
similarity_matrix = cosine_similarity(fv)
print(similarity_matrix)

# # Construct graph
# knn_graph = kneighbors_graph(fv, n_neighbors=3, metric='cosine')
# graph = nx.from_scipy_sparse_array(knn_graph)
#
# nx.draw(graph)
# plt.savefig("knngraph.png")
#
# saving_feature_vector(fv)

print('Time taken:', time.time() - st)

