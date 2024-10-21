import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.nn import Linear, Conv2d, MaxPool2d, BatchNorm2d
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image


def normalization_channel_wise(dataframe):
    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])
    base_ds = RGBImageDataset(dataframe=dataframe, mode='train', transforms=tf)
    loader = DataLoader(base_ds, batch_size=1, shuffle=True)

    total_images_count = [0, 0, 0, 0]
    mean = [0, 0, 0, 0]
    std = [0, 0, 0, 0]
    for images, _ in loader:
        for i, image in enumerate(images):
            image = image.view(image.size(0), image.size(1), -1)
            mean[i] += image.mean(2).sum(0)
            std[i] += image.std(2).sum(0)
            total_images_count[i] += image.size(0)
    for i in range(len(total_images_count)):
        mean[i] /= total_images_count[i]
        std[i] /= total_images_count[i]
        print(mean[i], std[i])


class RGBImageDataset(Dataset):
    def __init__(self, dataframe, img_dim=(224, 224), mode=None,
                 transforms=None, target_transforms=None):
        self.df = dataframe
        self.mode = mode
        self.dim = img_dim
        self.transforms = transforms
        self.target_transform = target_transforms

        if self.mode != 'predict':
            self.label_list = dataframe['label'].tolist()
            self.label_mapping = {'rbc': 0, 'platelet': 1, 'wbc platelet': 2, 'wbc': 3}
            self.label_list = [self.label_mapping[label] for label in self.label_list]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = []
        for channel in ['chan2', 'chan3', 'chan7', 'chan11']:
            img = self.df[channel].iloc[idx]
            img = cv2.imread(img)
            img = cv2.resize(img, self.dim)
            if self.transforms:
                img = self.transforms(img)
            image.append(img)
        if self.mode != 'predict':
            label = self.label_list[idx]
            if self.target_transform:
                label = self.target_transform(label)
            return image, label
        else:
            return image


class RGBConvNet(nn.Module):
    def __init__(self, num_channels=3, img_size=224, batch_size=4, device=torch.device('cuda')):
        super().__init__()
        self.kernel_size = 6
        self.conv1 = nn.Conv2d(num_channels, 6, self.kernel_size)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(p=0.5)
        self.conv2 = nn.Conv2d(6, 12, self.kernel_size)
        self.bn2 = nn.BatchNorm2d(12)
        self.conv3 = nn.Conv2d(12, 16, self.kernel_size)
        self.bn3 = nn.BatchNorm2d(16)

        conv_output_size = self.get_post_flatten_dim(img_size, batch_size)
        self.fc1 = nn.Linear(conv_output_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 42)
        self.fc4 = nn.Linear(42, 4)
        self.softmax = nn.LogSoftmax(dim=1)
        self.device = device

    def get_post_flatten_dim(self, img_size, batch_size):
        dummy_input = torch.zeros(batch_size, 3, img_size, img_size)
        input_list = []
        for _ in range(4):
            inp = self.bn1(self.conv1(dummy_input))
            inp = self.pool(F.relu(inp))

            inp = self.bn2(self.conv2(inp))
            inp = self.pool(F.relu(inp))

            inp = self.bn3(self.conv3(inp))
            inp = self.pool(F.relu(inp))

            input_list.append(torch.flatten(inp, 1))  # flatten all dimensions except batch

        return int(torch.cat(input_list, dim=1).size(1))

    def forward(self, inp1, inp2, inp3, inp4):
        input_list = []
        stacked_input = [inp1, inp2, inp3, inp4]
        for input_img in stacked_input:
            inp = self.bn1(self.conv1(input_img))
            inp = self.pool(F.relu(inp))

            inp = self.bn2(self.conv2(inp))
            inp = self.pool(F.relu(inp))

            inp = self.bn3(self.conv3(inp))
            inp = self.pool(F.relu(inp))

            # print(f'Size at last convolutional layer {inp.size()}')
            inp = torch.flatten(inp, 1)
            # print(f'Flattening dimensions except batch: {inp.size()}')
            input_list.append(inp)  # flatten all dimensions except batch

        x = torch.cat(input_list, dim=1)
        # print(f'Size after concatenation: {x.size()}')

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        #
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        x = F.relu(self.fc3(x))
        x = self.dropout(x)

        x = self.fc4(x)
        return x

    def summary(self, inp1, inp2, inp3, inp4):
        input_list = []
        # example_input = torch.zeros((batch_size, 3, input_height, input_width))
        stacked_input = [inp1, inp2, inp3, inp4]
        for input_img in stacked_input:
            inp = self.bn1(self.conv1(input_img))
            print(self.conv1.__class__.__name__)
            print(inp.size())
            inp = self.pool(F.relu(inp))
            print(self.pool.__class__.__name__)
            print(inp.size())

            inp = self.bn2(self.conv2(inp))
            print(self.conv2.__class__.__name__)
            print(inp.size())
            inp = self.pool(F.relu(inp))
            print(self.pool.__class__.__name__)
            print(inp.size())

            inp = self.bn3(self.conv3(inp))
            print(self.conv3.__class__.__name__)
            print(inp.size())

            inp = self.pool(F.relu(inp))
            print(self.pool.__class__.__name__)
            print(inp.size())

            # print(f'Size at last convolutional layer {inp.size()}')
            inp = torch.flatten(inp, 1)
            print(torch.flatten.__class__.__name__)
            print(inp.size())
            # print(f'Flattening dimensions except batch: {inp.size()}')
            input_list.append(inp)  # flatten all dimensions except batch

        x = torch.cat(input_list, dim=1)
        print(f'Size after concatenation: {x.size()}')

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        print(x.size())
        #
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        print(x.size())

        x = F.relu(self.fc3(x))
        x = self.dropout(x)

        print(x.size())

        x = self.fc4(x)

        print(x.size())
        return x

