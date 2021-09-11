import torch
from torch.utils.data import Dataset


class MultiNormalDataset(Dataset):
    def __init__(self, size=10000, mean1=1, mean2=-1, std1=1, std2=1):
        self.size = size
        self.data = torch.cat([torch.normal(mean1, std1, (size//2, 1)), torch.normal(mean2, std2, (size//2, 1))], dim=0)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.size
