import torch
import numpy as np
import matplotlib.pyplot as plt
from rgb_cnn import RGBConvNet
from multicell_classification.cnn import ConvNet

device = torch.device('cuda')
example_input = torch.zeros((16, 3, 224, 224)).to(device)
example_model = RGBConvNet(num_channels=3, device=device, batch_size=16).to(device)
second_model = ConvNet(num_channels=3, img_size=224, batch_size=16).to(device)
example_model.eval()
second_model.eval()

# example_output = example_model.summary(example_input, example_input, example_input, example_input)

second_output = second_model.summary(example_input)