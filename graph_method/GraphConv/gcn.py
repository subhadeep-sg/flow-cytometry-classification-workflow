import torch
from torch_geometric.data import Dataset


class GraphDataset(Dataset):
    def __init__(self, feature_vector):
        super().__init__()
        self.feature_vector = feature_vector

    def __len__(self):
        return len(self.feature_vector.shape[0])

    def process(self):
        idx = 0

    def get(self, idx):
        
        return self.feature_vector
