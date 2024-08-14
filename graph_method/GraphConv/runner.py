import numpy as np
import pandas as pd
from color_classify.metrics import result_visualizer
from color_classify.feature_vectors import reading_feature_vector
from sklearn.metrics.pairwise import cosine_similarity
import itertools
from torch_geometric.data import Data

root_dir = 'C:/DATA/UGASem5/GNNReferenceCodes/GraphStructures/ColorBasedGraphs/'
# df = pd.read_csv(root_dir + 'filename_list.csv')
fv = reading_feature_vector(file=root_dir + 'featurevector.txt',
                            as_arr=True, dtype='float32', verbose=False)

sim = cosine_similarity(fv, fv)
temp_df = pd.DataFrame(fv)
nodes = np.array(temp_df.index.values)

"""
Graph Dataset:
Create a permutation list of all nodes with all other nodes
Nodes are labelled by number of rows in fv. (2201 rows)
Then, the distance between respective nodes,
is calculated using a distance measure of choice.
"""
#
# nodes = np.array(temp_df.index.values)[0:100]
# print(nodes)
#
# all_edges = np.array([]).reshape((0, 2))
#
# print('Building edge matrix:')
# for node in nodes:
#     permutations = list(itertools.permutations(nodes, 2))
#     source = [e[0] for e in permutations]
#     target = [e[1] for e in permutations]
#     current_edges = np.column_stack([source, target])
#     all_edges = np.vstack([all_edges, current_edges])
#
# edge_index = all_edges.transpose()
#
# print(all_edges.shape)
# print(edge_index)
# print(all_edges)

""" 
Closer a number is to 1.0, the more similar they are 
"""
scale_factor = 1000
new_arr = sim.copy() * scale_factor
new_arr[new_arr < (scale_factor-1)] = 0
new_arr[new_arr >= scale_factor] = 0

edge_index = np.transpose(np.nonzero(new_arr)).T
print(edge_index)
print(edge_index.shape)
# print(new_arr)
print(np.count_nonzero(new_arr))
# print(np.transpose(np.nonzero(new_arr)))

graph_data = Data(x=fv, edge_index=edge_index)


