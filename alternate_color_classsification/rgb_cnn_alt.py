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


class RGBConcatImageDataset(Dataset):
    def __init__(self, dataframe, img_dim=(224, 224), mode=None,
                 transforms=None, target_transforms=None):
        self.df = dataframe
        self.mode = mode
        self.dim = img_dim
        self.transforms = transforms
        self.target_transform = target_transforms

        if self.mode != 'predict':
            self.label_list = dataframe['label'].tolist()
            # self.label_mapping = {'rbc': 0, 'platelet': 1, 'wbc platelet': 2, 'wbc': 3}
            # self.label_list = [self.label_mapping[label] for label in self.label_list]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        images = []
        for channel in ['chan2', 'chan3', 'chan7', 'chan11']:
            img = cv2.imread(self.df[channel].iloc[idx])  # reads as BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.dim)
            images.append(img)

        # Concatenate side by side (along width): shape = (H, W * 4, 3)
        concat_img = np.concatenate(images, axis=1)

        # Convert to tensor (C, H, W)
        concat_img = transforms.ToTensor()(concat_img)

        if self.transforms:
            concat_img = self.transforms(concat_img)

        if self.mode != 'predict':
            label = self.label_list[idx]
            if self.target_transform:
                label = self.target_transform(label)
            return concat_img, label
        else:
            return concat_img


class RGBConvNetConcat(nn.Module):
    def __init__(self, in_channels=3, num_classes=4, img_height=224, img_width=896, *args, **kwargs):  # 896 = 4 * 224
        super(RGBConvNetConcat, self).__init__()
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(64)

        self.dropout = nn.Dropout(0.5)


        dummy_input = torch.zeros(1, in_channels, img_height, img_width)
        x = self._forward_features(dummy_input)
        flattened_dim = x.view(1, -1).size(1)

        self.fc1 = nn.Linear(flattened_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def _forward_features(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # -> (B, 16, H1, W1)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # -> (B, 32, H2, W2)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # -> (B, 64, H3, W3)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


    # def summary(self, inp1, inp2, inp3, inp4):
    #     input_list = []
    #     # example_input = torch.zeros((batch_size, 3, input_height, input_width))
    #     stacked_input = [inp1, inp2, inp3, inp4]
    #     for input_img in stacked_input:
    #         inp = self.bn1(self.conv1(input_img))
    #         print(self.conv1.__class__.__name__)
    #         print(inp.size())
    #         inp = self.pool(F.relu(inp))
    #         print(self.pool.__class__.__name__)
    #         print(inp.size())
    #
    #         inp = self.bn2(self.conv2(inp))
    #         print(self.conv2.__class__.__name__)
    #         print(inp.size())
    #         inp = self.pool(F.relu(inp))
    #         print(self.pool.__class__.__name__)
    #         print(inp.size())
    #
    #         inp = self.bn3(self.conv3(inp))
    #         print(self.conv3.__class__.__name__)
    #         print(inp.size())
    #
    #         inp = self.pool(F.relu(inp))
    #         print(self.pool.__class__.__name__)
    #         print(inp.size())
    #
    #         # print(f'Size at last convolutional layer {inp.size()}')
    #         inp = torch.flatten(inp, 1)
    #         print(torch.flatten.__class__.__name__)
    #         print(inp.size())
    #         # print(f'Flattening dimensions except batch: {inp.size()}')
    #         input_list.append(inp)  # flatten all dimensions except batch
    #
    #     x = torch.cat(input_list, dim=1)
    #     print(f'Size after concatenation: {x.size()}')
    #
    #     x = F.relu(self.fc1(x))
    #     x = self.dropout(x)
    #
    #     print(x.size())
    #     #
    #     x = F.relu(self.fc2(x))
    #     x = self.dropout(x)
    #
    #     print(x.size())
    #
    #     x = F.relu(self.fc3(x))
    #     x = self.dropout(x)
    #
    #     print(x.size())
    #
    #     x = self.fc4(x)
    #
    #     print(x.size())
    #     return x
    #
