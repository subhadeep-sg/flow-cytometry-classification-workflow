import numpy as np
from torch.utils.data import Dataset
import os
import cv2
# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

#
# class ImageDataset(Dataset):
#     def __init__(self, img_dir, mode=None, img_dimensions=(224, 224),
#                  transforms=None, target_transforms=None):
#         self.mode = mode
#         self.dim = img_dimensions
#         if self.mode in ['train', 'val', 'test']:
#             self.img_dir = img_dir + '/' + self.mode
#         else:
#             self.img_dir = img_dir
#         self.transform = transforms
#         self.target_transform = target_transforms
#         self.filelist = [path.replace("\\", "/") + '/' + file for path, _, filenames in os.walk(self.img_dir) for file
#                          in filenames]
#         if self.mode != 'predict':
#             self.label_list = [path.split('\\')[-1] for path, _, filenames in os.walk(self.img_dir) for file in
#                                filenames]
#             self.class_names = ['particles', 'single', 'multi']
#             self.label_mapping = {'particles': 0, 'single': 1, 'multi': 2}
#             self.label_list = [self.label_mapping[label] for label in self.label_list]
#
#     def __len__(self):
#         return len(self.filelist)
#
#     def __getitem__(self, idx):
#         image_path = self.filelist[idx]
#         image = cv2.imread(image_path)
#         image = cv2.resize(image, self.dim)
#         if self.transform:
#             image = self.transform(image)
#
#         if self.mode != 'predict':
#             label = self.label_list[idx]
#             if self.target_transform:
#                 label = self.target_transform(label)
#             return image, label
#         else:
#             label = None
#             return image
