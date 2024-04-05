import tensorflow as tf
from tensorflow.python.client import device_lib
import torch

import torch
print(f'PyTorch version: {torch.__version__}')
print('*'*10)
print('*'*10)
print(f'CUDNN version: {torch.backends.cudnn.version()}')
print(f'Available GPU devices: {torch.cuda.device_count()}')
print(f'Device Name: {torch.cuda.get_device_name()}')