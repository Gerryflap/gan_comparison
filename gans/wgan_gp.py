import torch
from torch.optim import Adam

from gans.abstract_gan import AbstractGan


class WGanGP(AbstractGan):
    """
        Wasserstein GAN with Gradient Penalty https://arxiv.org/pdf/1704.00028.pdf
    """

    def __init__(self, h_size, z_size, n_features=1, learning_rate=1e-3, lambd=10.0, use_batchnorm=False, G_step_every=5):
        super().__init__(h_size, z_size, n_features=n_features, learning_rate=learning_rate, use_batchnorm=False)
        self.lambd = lambd
        self.step = 0
        self.G_step_every = G_step_every

        if use_batchnorm:
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

        # WGAN-GP uses alternative betas to stabilize training
        self.opt_G = Adam(self.G.parameters(), lr=learning_rate, betas=(0.0, 0.9))
        self.opt_D = Adam(self.D.parameters(), lr=learning_rate, betas=(0.0, 0.9))

    def _train_step(self, real_batch):
        # Train D
        generated_batch = self.generate_batch(real_batch.size(0)).detach()

        self.opt_D.zero_grad()

        pred_real = self.D(real_batch)
        pred_fake = self.D(generated_batch)

        epsilon = torch.rand_like(pred_real)
        x_hat = real_batch * epsilon + generated_batch * (1.0 - epsilon)
        x_hat.requires_grad = True
        pred_xhat = self.D(x_hat)
        grad_outputs = torch.ones_like(pred_xhat)
        gradients = torch.autograd.grad(pred_xhat, x_hat, create_graph=True, only_inputs=True, retain_graph=True,
                                        grad_outputs=grad_outputs)[0]
        # gp = torch.square(torch.norm(gradients, 2, dim=1) - 1.0)
        gp = torch.square(torch.clip(torch.norm(gradients, 2, dim=1), min=1.0) - 1.0)

        loss_D = (pred_fake - pred_real + self.lambd * gp).mean()

        loss_D.backward()
        self.opt_D.step()

        if self.step % self.G_step_every == 0:
            # Train G
            generated_batch = self.generate_batch(real_batch.size(0))
            pred_fake = self.D(generated_batch)

            loss_G = self._generator_loss(pred_fake)
            self.opt_G.zero_grad()
            loss_G.backward()
            self.opt_G.step()
        self.step += 1

    def _generator_loss(self, pred_fake):
        return -torch.mean(pred_fake)
