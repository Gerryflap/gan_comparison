import torch
from torch.utils.data import Dataset


class MultiModeDataset(Dataset):
    def __init__(self, means: list, size_per_mode=10000, stddev=1.0):
        self.size = size_per_mode * len(means)
        modes = []
        for mean in means:
            modes.append(torch.normal(mean, stddev, (size_per_mode, 1)))
        self.data = torch.cat(modes, dim=0)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.size
