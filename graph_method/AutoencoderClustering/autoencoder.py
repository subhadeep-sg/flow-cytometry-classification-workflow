import torch
from torch.nn import Linear, ReLU, Sigmoid, Sequential
from torch.utils.data import Dataset
import PIL
from skimage import io, transform


class FeatureVecDataset(Dataset):
    def __init__(self, feature_vector, dataframe=None, trans=None):
        self.feature_vector = feature_vector
        self.df = dataframe
        self.transform = trans

    def __len__(self):
        return self.feature_vector.shape[0]

    def __getitem__(self, item):
        fv = self.feature_vector[item]
        fv = torch.tensor(fv)
        data = fv
        # image = self.df['chan2'][item]
        # image = io.imread(image)
        # if self.transform:
        #     data['image'] = self.transform(data['image'])
        return data


class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Sequential(
            Linear((256*4), 256),
            ReLU(),
            Linear(256, 128),
            ReLU(),
            Linear(128, 64),
            ReLU(),
            Linear(64, 36)
            # ReLU(),
            # Linear(36, 18)
        )

        self.decoder = Sequential(
            # Linear(18, 36),
            # ReLU(),
            Linear(36, 64),
            ReLU(),
            Linear(64, 128),
            ReLU(),
            Linear(128, 256),
            ReLU(),
            Linear(256, (256*4)),
            Sigmoid())

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
