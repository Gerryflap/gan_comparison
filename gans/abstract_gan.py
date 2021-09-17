from abc import ABC, abstractmethod
import torch


class AbstractGan(ABC, torch.nn.Module):
    def __init__(self, h_size, z_size, learning_rate=1e-3, n_features=1, use_batchnorm=False):
        super().__init__()
        self.h_size = h_size
        self.z_size = z_size
        self.use_batchnorm = use_batchnorm

        if self.use_batchnorm:
            self.G = torch.nn.Sequential(
                torch.nn.Linear(z_size, h_size),
                torch.nn.BatchNorm1d(h_size),
                torch.nn.ReLU(),

                torch.nn.Linear(h_size, h_size),
                torch.nn.BatchNorm1d(h_size),
                torch.nn.ReLU(),

                torch.nn.Linear(h_size, h_size),
                torch.nn.BatchNorm1d(h_size),
                torch.nn.ReLU(),

                torch.nn.Linear(h_size, n_features)
            )

            self.D = torch.nn.Sequential(
                torch.nn.Linear(n_features, h_size),
                # Batchnorm here might cause instability
                torch.nn.LeakyReLU(),

                torch.nn.Linear(h_size, h_size),
                torch.nn.BatchNorm1d(h_size),
                torch.nn.LeakyReLU(),

                torch.nn.Linear(h_size, h_size),
                torch.nn.BatchNorm1d(h_size),
                torch.nn.LeakyReLU(),

                torch.nn.Linear(h_size, 1)
            )
        else:
            self.G = torch.nn.Sequential(
                torch.nn.Linear(z_size, h_size),
                torch.nn.ReLU(),

                torch.nn.Linear(h_size, h_size),
                torch.nn.ReLU(),

                torch.nn.Linear(h_size, h_size),
                torch.nn.ReLU(),

                torch.nn.Linear(h_size, n_features)
            )

            self.D = torch.nn.Sequential(
                torch.nn.Linear(n_features, h_size),
                torch.nn.LeakyReLU(),

                torch.nn.Linear(h_size, h_size),
                torch.nn.LeakyReLU(),

                torch.nn.Linear(h_size, h_size),
                torch.nn.LeakyReLU(),

                torch.nn.Linear(h_size, 1)
            )

        self.opt_G = torch.optim.Adam(self.G.parameters(), learning_rate)
        self.opt_D = torch.optim.Adam(self.D.parameters(), learning_rate)

    def train_step(self, real_batch):
        self.train()
        if not real_batch.is_cuda:
            device = next(self.G.parameters()).device
            real_batch = real_batch.to(device)
        self._train_step(real_batch)
        self.eval()

    @abstractmethod
    def _train_step(self, real_batch):
        pass

    def generate_batch(self, batch_size, train=True):
        if train:
            self.train()
        else:
            self.eval()
        device = next(self.G.parameters()).device
        z = torch.normal(0, 1, (batch_size, self.z_size), device=device)

        gen_x = self.G(z)
        return gen_x

    def get_discriminator_values_1d(self, range=(-4, 4), steps=100):
        with torch.no_grad():
            device = next(self.G.parameters()).device
            inp = torch.linspace(range[0], range[1], steps=steps, device=device, dtype=torch.float32).view(-1, 1)
            return inp.view(-1), self.D(inp).view(-1)
