from sklearn.decomposition import PCA
import pandas as pd
import time
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
import numpy as np
from color_classify.feature_vectors import reading_feature_vector
from color_classify.metrics import result_visualizer

st = time.time()

image_csv = 'filename_list.csv'
df = pd.read_csv(image_csv)
fv = np.array(reading_feature_vector('featurevector.txt', verbose=False))

# kmeans = KMeans(n_clusters=3, random_state=0).fit(fv)
# kmeans_labels = kmeans.labels_
#
# gmm = GaussianMixture(n_components=3, random_state=0).fit(fv)
# gmm_labels = gmm.predict(fv)

# result_visualizer(df, kmeans_labels, model_name='Kmeans', multi_channels=True)
# result_visualizer(df, gmm_labels, model_name='GMM', multi_channels=True)

# PCA
# pca = PCA(n_components=128)
# x = pca.fit_transform(fv)
#
# km_pca = KMeans(n_clusters=3, random_state=0).fit(x)
# km_pca_labels = km_pca.labels_

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(fv)

# Apply K-means on the reduced data
km_tsne = KMeans(n_clusters=3, random_state=0).fit(X_tsne)
km_tsne_labels = km_tsne.labels_

# result_visualizer(df, km_pca_labels, model_name='KMeans+PCA', multi_channels=True, num_samples=20)
result_visualizer(df, km_tsne_labels, model_name='KMeans+TSNE', multi_channels=True, num_samples=8)

print('Time taken: ', time.time() - st)
