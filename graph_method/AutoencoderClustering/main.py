from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

from autoencoder import FeatureVecDataset, AutoEncoder
import matplotlib.pyplot as plt
import torch
import pandas as pd

from color_classification.color_classify.metrics import result_visualizer, cluster_scoring
from color_classification.color_classify.feature_vectors import reading_feature_vector
from torch.utils.data import DataLoader
import time
import numpy as np

st = time.time()
root_dir = '../../graph_method/'
# df = pd.read_csv(root_dir + 'filename_list.csv')
fv = reading_feature_vector(file=root_dir + 'featurevector.txt',
                            as_arr=True, dtype='float32', verbose=False)

norm = np.linalg.norm(fv)
normalized_array = fv / norm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
fv_dataset = FeatureVecDataset(feature_vector=normalized_array)
loader = DataLoader(dataset=fv_dataset, batch_size=1, shuffle=True)
print('Loader shape: ', len(loader))

model = AutoEncoder().to(device=device)
loss_fn = torch.nn.MSELoss()
lr_factor = 1
optimizer = torch.optim.Adam(model.parameters(),
                             lr=0.001 * lr_factor,
                             weight_decay=1e-4)     # 1e-8

epochs = 100     # 20
outputs = []
losses = []
train_time = time.time()
print('Training start...')
for ep in range(epochs):
    print('Epoch: {}'.format(ep))
    for data in loader:
        data = data.to(device)
        model_output = model(data)

        # print(f'data shape: {data.shape}, model_output shape: {model_output.shape}')
        # print(f'data range: [{data.min()}, {data.max()}]')
        # print(f'model_output range: [{model_output.min()}, {model_output.max()}]')

        loss = loss_fn(model_output, data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    outputs.append((ep, data, model_output))
print("Training time: {}".format(time.time() - train_time))

print("Model output shape: ", model_output.shape)

plt.style.use('fivethirtyeight')
plt.xlabel('Iterations')
plt.ylabel('Loss')
# plt.plot(losses[::-1])
plt.plot(losses)
plt.show()

# Inference step to get prediction from trained model
predictions = []
fmaps = []
with torch.no_grad():
    for dat in loader:
        dat = dat.to(device)
        predictions.append(model(dat).tolist()[0])
        mapping = model.encoder(dat).tolist()[0]
        fmaps.append(mapping)

predictions = np.array(predictions)
print(predictions)

image_csv = '../../color_classification/groundtruths.csv'
df = pd.read_csv(image_csv, index_col=False)
labels = df['label'].tolist()
labels = LabelEncoder().fit_transform(labels)

# Clustering on the predicted data
km_ae = KMeans(n_clusters=3, random_state=0).fit(predictions)                   #predictions)
km_ae_labels = km_ae.labels_
cluster_scoring(labels, km_ae_labels, 'Autoencoder + Kmeans')

# image_csv = root_dir + 'filename_list.csv'
# df = pd.read_csv(image_csv)
#
# df = df.replace({'../MasterDataset': 'C:/DATA/UGASem5/FlowCytometry/flow-cytometry-classification-workflow'
#                                      '/MasterDataset'}, regex=True)
#
# print(len(df))

# result_visualizer(df, km_ae_labels, model_name='KMeans+AE', multi_channels=True, num_samples=8)

print("Total runtime: {}".format(time.time() - st))
