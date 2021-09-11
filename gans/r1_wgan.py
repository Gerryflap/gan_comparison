import torch

from gans.abstract_gan import AbstractGan


class R1WGan(AbstractGan):
    """
        Combination of WGAN objective and R1 regularization (gradient penalty) https://arxiv.org/abs/1801.04406v4
    """

    def __init__(self, h_size, z_size, n_features=1, learning_rate=1e-3, gamma=10.0):
        super().__init__(h_size, z_size, n_features=n_features, learning_rate=learning_rate)
        self.gamma = gamma

    def _train_step(self, real_batch):
        # Train D
        generated_batch = self.generate_batch(real_batch.size(0)).detach()
        generated_batch.requires_grad = True

        pred_real = self.D(real_batch)
        pred_fake = self.D(generated_batch)

        loss_D = -(torch.mean(pred_real) - torch.mean(pred_fake))

        gradients = torch.autograd.grad(pred_fake.sum(), generated_batch, create_graph=True, only_inputs=True)[0]
        r1_regularization = 0.5 * torch.mean(torch.norm(gradients, 2, 1))
        loss_D = loss_D + self.gamma * r1_regularization

        self.opt_D.zero_grad()
        loss_D.backward()
        self.opt_D.step()

        # Train G
        generated_batch = self.generate_batch(real_batch.size(0))
        pred_fake = self.D(generated_batch)

        loss_G = self._generator_loss(pred_fake)
        self.opt_G.zero_grad()
        loss_G.backward()
        self.opt_G.step()

    def _generator_loss(self, pred_fake):
        return -torch.mean(pred_fake)
