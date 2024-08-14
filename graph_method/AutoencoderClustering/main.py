from sklearn.cluster import KMeans

from autoencoder import FeatureVecDataset, AutoEncoder
import matplotlib.pyplot as plt
import torch
import pandas as pd

from color_classify.metrics import result_visualizer
from color_classify.feature_vectors import reading_feature_vector
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import time
import numpy as np

st = time.time()
root_dir = 'C:/DATA/UGASem5/GNNReferenceCodes/GraphStructures/ColorBasedGraphs/'
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
optimizer = torch.optim.Adam(model.parameters(),
                             lr=1e-2,
                             weight_decay=1e-8)

epochs = 20
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
with torch.no_grad():
    for dat in loader:
        dat = dat.to(device)
        predictions.append(model(dat).tolist()[0])

predictions = np.array(predictions)
print(predictions)

# Clustering on the predicted data
km_ae = KMeans(n_clusters=3, random_state=0).fit(predictions)
km_ae_labels = km_ae.labels_

image_csv = root_dir + 'filename_list.csv'
df = pd.read_csv(image_csv)
result_visualizer(df, km_ae_labels, model_name='KMeans+AE', multi_channels=True, num_samples=8)

print("Total runtime: {}".format(time.time() - st))
