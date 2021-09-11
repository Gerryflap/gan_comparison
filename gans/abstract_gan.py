from abc import ABC, abstractmethod
import torch


class AbstractGan(ABC, torch.nn.Module):
    def __init__(self, h_size, z_size, learning_rate=1e-3):
        super().__init__()
        self.h_size = h_size
        self.z_size = z_size

        self.G = torch.nn.Sequential(
            torch.nn.Linear(z_size, h_size),
            torch.nn.ReLU(),

            torch.nn.Linear(h_size, h_size),
            torch.nn.ReLU(),

            torch.nn.Linear(h_size, 1)
        )

        self.D = torch.nn.Sequential(
            torch.nn.Linear(1, h_size),
            torch.nn.LeakyReLU(),

            torch.nn.Linear(h_size, h_size),
            torch.nn.LeakyReLU(),

            torch.nn.Linear(h_size, 1)
        )

        self.opt_G = torch.optim.Adam(self.G.parameters(), learning_rate)
        self.opt_D = torch.optim.Adam(self.D.parameters(), learning_rate)

    def train_step(self, real_batch):
        device = next(self.G.parameters()).device
        real_batch = real_batch.to(device)
        self._train_step(real_batch)

    @abstractmethod
    def _train_step(self, real_batch):
        pass

    def generate_batch(self, batch_size):
        device = next(self.G.parameters()).device
        z = torch.normal(0, 1, (batch_size, self.z_size), device=device)

        gen_x = self.G(z)
        return gen_x
