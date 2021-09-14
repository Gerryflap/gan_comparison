import math

import torch
from torch.utils.data import Dataset


class CircleOfGaussiansDataset2D(Dataset):
    def __init__(self, size=10000, n_gaussians=10, stddev=0.1):
        self.size = size

        sets = []
        for i in range(n_gaussians):
            angle = (i/n_gaussians) * 2.0 * math.pi
            mean_x, mean_y = math.cos(angle), math.sin(angle)
            sets.append(torch.cat([
                torch.normal(mean_x, stddev, (size // n_gaussians, 1)),
                torch.normal(mean_y, stddev, (size // n_gaussians, 1))
            ], dim=1))

        self.data = torch.cat(sets, dim=0)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.size(0)
