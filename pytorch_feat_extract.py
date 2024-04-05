import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_loading import DataLoad
import os


import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split, default_collate
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# os.environ["TP_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# AUTOTUNE = tf.data.AUTOTUNE
batch_size = 2
img_height = 224
img_width = 224

"""
This file attempts to generate feature extraction pipeline using PyTorch using dataloading and then 
defining a simple neural network
"""

df = pd.read_csv('small.csv')


# data = DataLoad(main_path='MasterDataset/', dim=(img_height, img_width))
# data.get_config('categories.json')
# f = data.get_dataframe()

# A rough instance to check for updates
# experiment = DataLoad(main_path='MasterDataset/', dim=(224, 224))

# img_arrays = df['image_array'].to_list()
# img_names = df['image_name'].to_list()
# img_labels = df['class_label'].values
# class_names = df['class_label'].unique()

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(24)
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(24 * 10 * 10, 10)

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.pool(output)
        output = F.relu(self.bn4(self.conv4(output)))
        output = F.relu(self.bn5(self.conv5(output)))
        output = output.view(-1, 24 * 10 * 10)
        output = self.fc1(output)

        return output


# Instantiate a neural network model
model = Network()


class MyDataset(Dataset):
    def __init__(self, dataframe, transform_=None):
        self.data = dataframe
        self.transform = transform_
        # self.img_arrays = df['image_array'].to_list()
        # self.img_names = df['image_name'].to_list()
        # self.img_labels = df['class_label'].values
        # self.class_names = df['class_label'].unique()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = self.data.image_name[index]
        # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = Image.open(image_path)  # .convert('L')
        label = self.data.class_label[index]
        if self.transform:
            image = self.transform(image)
        return image, label


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    # transforms.ToPILImage(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
    #                      std=[0.2023, 0.1994, 0.2010]),
])

dataset = MyDataset(df, transform)
trainset, testset = random_split(dataset, [0.7, 0.3])
dataloader = DataLoader(trainset, batch_size=2, shuffle=True)

# model = nn.Sequential(
#     nn.Linear(224, 60),
#     nn.ReLU(),
#     nn.Linear(60, 30),
#     nn.ReLU(),
#     nn.Linear(30, 1),
#     nn.Sigmoid()
# )

# Train the model
n_epochs = 10
loss_fn = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
model.train()
for epoch in range(n_epochs):
    for X_batch, y_batch in dataloader:
        y_pred = model(X_batch)
        print(y_pred.shape)
        print(y_batch.shape)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# evaluate accuracy after training
X_test, y_test = default_collate(testset)
model.eval()
y_pred = model(X_test)
acc = (y_pred.round() == y_test).float().mean()
acc = float(acc)
print("Model accuracy: %.2f%%" % (acc * 100))
