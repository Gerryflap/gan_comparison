import torch
from torch.optim import RMSprop

from gans.abstract_gan import AbstractGan


class R1Gan(AbstractGan):
    """
        Non-saturating GAN with R1 regularization (gradient penalty) https://arxiv.org/abs/1801.04406v4
    """

    def __init__(self, h_size, z_size, n_features=1, learning_rate=1e-3, gamma=10.0, non_saturating=True, use_batchnorm=False):
        super().__init__(h_size, z_size, n_features=n_features, learning_rate=learning_rate, use_batchnorm=use_batchnorm)
        self.gamma = gamma
        self.ns = non_saturating

        # As described in the paper
        self.opt_G = RMSprop(self.G.parameters(), lr=learning_rate, alpha=0.9)
        self.opt_D = RMSprop(self.D.parameters(), lr=learning_rate, alpha=0.9)



    def _train_step(self, real_batch):
        # Train D
        generated_batch = self.generate_batch(real_batch.size(0)).detach()
        generated_batch.requires_grad = True

        pred_real = torch.sigmoid(self.D(real_batch))
        pred_fake = torch.sigmoid(self.D(generated_batch))

        grad_outputs = torch.ones_like(pred_fake)
        gradients = torch.autograd.grad(pred_fake, generated_batch, create_graph=True, only_inputs=True,
                                        retain_graph=True, grad_outputs=grad_outputs)[0]
        r1_regularization = 0.5 * torch.norm(gradients, 2, 1)**2.0
        loss_D = -(torch.log(pred_real + 1e-6) + (torch.log(1.0 - pred_fake + 1e-6))) + self.gamma * r1_regularization
        loss_D = loss_D.mean()
        generated_batch.requires_grad = False

        self.opt_D.zero_grad()
        loss_D.backward()
        self.opt_D.step()

        # Train G
        generated_batch = self.generate_batch(real_batch.size(0))
        pred_fake = torch.sigmoid(self.D(generated_batch))

        loss_G = self._generator_loss(pred_fake)
        self.opt_G.zero_grad()
        loss_G.backward()
        self.opt_G.step()

    def _generator_loss(self, pred_fake):
        if self.ns:
            return -torch.mean(torch.log(pred_fake + 1e-6))
        else:
            return torch.mean(torch.log(1.0 - pred_fake + 1e-6))

    def get_discriminator_values_1d(self, range=(-4, 4), steps=100):
        with torch.no_grad():
            inp, values = super().get_discriminator_values_1d(range, steps)
            return inp, torch.sigmoid(values)