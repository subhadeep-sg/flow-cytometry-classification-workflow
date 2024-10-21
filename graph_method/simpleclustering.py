import pandas as pd
import time
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from sklearn import metrics
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder
from color_classification.color_classify.feature_vectors import reading_feature_vector
from color_classification.color_classify.metrics import result_visualizer, cluster_scoring

st = time.time()

image_csv = '../color_classification/groundtruths.csv'
df = pd.read_csv(image_csv, index_col=False)
labels = df['label'].tolist()
labels = LabelEncoder().fit_transform(labels)

fv = np.array(reading_feature_vector('featurevector.txt', verbose=False))

kmeans = KMeans(n_clusters=3, random_state=0).fit(fv)
kmeans_labels = kmeans.labels_
cluster_scoring(labels, kmeans_labels, 'Kmeans')
# result_visualizer(df, kmeans_labels, model_name='Kmeans', multi_channels=True)


gmm = GaussianMixture(n_components=3, random_state=0).fit(fv)
gmm_labels = gmm.predict(fv)
cluster_scoring(labels, gmm_labels, 'Gaussian Mixture')
# result_visualizer(df, gmm_labels, model_name='GMM', multi_channels=True)

# PCA
pca = PCA(n_components=128)         # n_components=128
x = pca.fit_transform(fv)

km_pca = KMeans(n_clusters=3, random_state=0).fit(x)
km_pca_labels = km_pca.labels_
cluster_scoring(labels, km_pca_labels, 'Kmeans + PCA')
# result_visualizer(df, km_pca_labels, model_name='KMeans+PCA', multi_channels=True, num_samples=20)

tsne = TSNE(n_components=50)        # n_components=2
X_tsne = tsne.fit_transform(fv)

km_tsne = KMeans(n_clusters=3, random_state=0).fit(X_tsne)
km_tsne_labels = km_tsne.labels_
cluster_scoring(labels, km_tsne_labels, 'Kmeans + TSNE')
#
#
# result_visualizer(df, km_tsne_labels, model_name='KMeans+TSNE', multi_channels=True, num_samples=8)

print('Time taken: ', time.time() - st)
