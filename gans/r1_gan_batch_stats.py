import torch
from torch.optim import Adam

from gans.abstract_gan import AbstractGan
from gans.r1_gan import R1Gan


class BatchStatDiscriminator(torch.nn.Module):
    def __init__(self, h_size, n_features=1):
        super().__init__()
        self.D_in = torch.nn.Sequential(
            torch.nn.Linear(n_features, h_size),
            torch.nn.LeakyReLU(),

            torch.nn.Linear(h_size, h_size),
            torch.nn.LeakyReLU(),
        )

        self.D_out = torch.nn.Sequential(
            torch.nn.Linear(h_size + 1, h_size),
            torch.nn.LeakyReLU(),

            torch.nn.Linear(h_size, 1)
        )

    def forward(self, inp):
        x = self.D_in(inp)
        std = torch.std(x[:, 0:1]).view(1, 1).expand((x.size(0), 1))
        x = torch.cat([std, x], dim=1)
        return self.D_out(x)


class R1GanBatchStats(R1Gan):
    """
        Non-saturating GAN with R1 regularization (gradient penalty) https://arxiv.org/abs/1801.04406v4
        and batch statistics (stddev of 1 feature in D ast input to the next layer)
    """

    def __init__(self, h_size, z_size, n_features=1, learning_rate=1e-3, gamma=10.0, non_saturating=True,
                 use_batchnorm=False):
        super().__init__(h_size, z_size, n_features=n_features, learning_rate=learning_rate, gamma=gamma,
                         use_batchnorm=use_batchnorm, non_saturating=non_saturating)

        self.D = BatchStatDiscriminator(h_size, n_features)
        self.opt_D = Adam(self.D.parameters(), learning_rate)
