import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
import os
import cv2
import warnings

warnings.filterwarnings('ignore')


class ImageDataset(Dataset):
    def __init__(self, img_dir=None, dataframe=None, mode=None, img_dimensions=(224, 224),
                 transforms=None, target_transforms=None):
        self.mode = mode
        assert img_dir is None or dataframe is None, "Please provide either dataframe or a directory!"
        self.df = dataframe
        self.dim = img_dimensions
        if self.mode in ['train', 'val', 'test'] and img_dir:
            self.img_dir = img_dir + '/' + self.mode
            self.filelist = [path.replace("\\", "/") + '/' + file for path, _, filenames in os.walk(self.img_dir) for
                             file in filenames]
        else:
            self.img_dir = img_dir
            self.filelist = self.df['filename'].tolist()
        self.transform = transforms
        self.target_transform = target_transforms

        self.label_mapping = {'particles': 0, 'single': 1, 'multi': 2}

        if self.df is not None and self.mode != 'predict':
            self.label_list = self.df['class_label'].tolist()
            self.label_list = [self.label_mapping[label] for label in self.label_list]
        elif self.mode != 'predict':
            self.label_list = [path.split('\\')[-1] for path, _, filenames in os.walk(self.img_dir) for file in
                               filenames]
            self.class_names = ['particles', 'single', 'multi']
            self.label_list = [self.label_mapping[label] for label in self.label_list]

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        if self.df is not None:
            image = self.filelist[idx]
            image = cv2.imread(image)
            image = cv2.resize(image, self.dim)
            if self.transform:
                image = self.transform(image)

        else:
            image_path = self.filelist[idx]
            image = cv2.imread(image_path)
            image = cv2.resize(image, self.dim)
            if self.transform:
                image = self.transform(image)

        if self.mode != 'predict':
            label = self.label_list[idx]
            if self.target_transform:
                label = self.target_transform(label)
            return image, label
        else:
            label = None
            return image

    def get_image_list(self):
        return self.filelist


class ConvNet(nn.Module):
    def __init__(self, num_channels=3, img_size=224, batch_size=4, device=torch.device('cuda')):
        super().__init__()
        self.kernel_size = 6
        self.conv1 = nn.Conv2d(num_channels, 6, self.kernel_size)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(p=0.5)
        self.conv7 = nn.Conv2d(6, 12, self.kernel_size)
        self.bn7 = nn.BatchNorm2d(12)

        self.conv2 = nn.Conv2d(12, 16, self.kernel_size)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, self.kernel_size)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, self.kernel_size)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, self.kernel_size)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 256, self.kernel_size)
        self.bn6 = nn.BatchNorm2d(256)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(2)
        conv_output_size = self.get_post_flatten_dim(img_size, batch_size)
        self.fc1 = nn.Linear(conv_output_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 42)
        self.fc4 = nn.Linear(42, 3)
        self.softmax = nn.LogSoftmax(dim=1)
        self.device = device

    def get_post_flatten_dim(self, img_size, batch_size):
        dummy_input = torch.zeros(batch_size, 3, img_size, img_size)
        x = self.pool(F.relu(self.conv1(dummy_input)))
        x = self.pool(F.relu(self.conv7(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten the output
        return int(torch.flatten(x, 1).size(1))

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.pool(F.relu(x))

        x = self.bn7(self.conv7(x))
        x = self.pool(F.relu(x))

        x = self.bn2(self.conv2(x))
        x = self.pool(F.relu(x))

        x = self.bn3(self.conv3(x))
        x = self.pool(F.relu(x))

        x = torch.flatten(x, 1)  # flatten all dimensions except batch

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        #
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        x = F.relu(self.fc3(x))
        x = self.dropout(x)

        x = self.fc4(x)
        # x = self.softmax(x)

        return x

    def summary(self, x):

        x = self.bn1(self.conv1(x))
        print(self.conv1.__class__.__name__)
        print(x.size())

        x = self.pool(F.relu(x))
        print(self.pool.__class__.__name__)
        print(x.size())

        x = self.bn7(self.conv7(x))
        print(self.conv7.__class__.__name__)
        print(x.size())

        x = self.pool(F.relu(x))
        print(self.pool.__class__.__name__)
        print(x.size())

        x = self.bn2(self.conv2(x))
        print(self.conv2.__class__.__name__)
        print(x.size())

        x = self.pool(F.relu(x))
        print(self.pool.__class__.__name__)
        print(x.size())

        x = self.bn3(self.conv3(x))
        print(self.conv3.__class__.__name__)
        print(x.size())

        x = self.pool(F.relu(x))
        print(self.pool.__class__.__name__)
        print(x.size())

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        print(x.size())

        x = F.relu(self.fc1(x))
        print(self.fc1.__class__.__name__)
        x = self.dropout(x)
        print(x.size())

        #
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        print(self.fc2.__class__.__name__)
        print(x.size())

        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        print(self.fc3.__class__.__name__)
        print(x.size())

        x = self.fc4(x)
        print(self.fc4.__class__.__name__)
        print(x.size())

        # x = self.softmax(x)

        return x

