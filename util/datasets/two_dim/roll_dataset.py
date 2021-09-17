import math

import torch
from torch.utils.data import Dataset


class RollDataset2D(Dataset):
    def __init__(self, size=10000, n_times_round=3, stddev=0.1):
        self.size = size

        deviation = torch.normal(0, stddev, (size, 1), dtype=torch.float32)
        angle = torch.linspace(0.0, math.pi * n_times_round * 2.0, size, dtype=torch.float32).view(-1, 1)
        dist = torch.linspace(0.0, 1.0, size, dtype=torch.float32).view(-1, 1)

        sin_angle, cos_angle = torch.sin(angle), torch.cos(angle)
        x_pos = cos_angle * dist - sin_angle * deviation
        y_pos = sin_angle * dist + cos_angle * deviation

        self.data = torch.cat([x_pos, y_pos], dim=1)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.size(0)
