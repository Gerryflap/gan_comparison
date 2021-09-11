import torch
from torch.utils.data import Dataset


class NormalRandomDataset(Dataset):
    def __init__(self, size=10000, mean=0, stddev=1):
        self.size = size
        self.data = torch.normal(mean, stddev, (size, 1))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.size
