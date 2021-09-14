import torch
from torch.optim import RMSprop

from gans.abstract_gan import AbstractGan


class WGan(AbstractGan):
    """
        Wasserstein GAN https://arxiv.org/pdf/1701.07875.pdf
    """

    def __init__(self, h_size, z_size, n_features=1, learning_rate=1e-3, clip=0.01):
        super().__init__(h_size, z_size, n_features=n_features, learning_rate=learning_rate, use_batchnorm=False)
        self.clip = clip
        self.step = 0

        # WGAN uses RMSProp, which supposedly influences the stability
        self.opt_G = RMSprop(self.G.parameters(), lr=learning_rate)
        self.opt_D = RMSprop(self.D.parameters(), lr=learning_rate)

    def _train_step(self, real_batch):
        # Train D
        generated_batch = self.generate_batch(real_batch.size(0)).detach()

        pred_real = self.D(real_batch)
        pred_fake = self.D(generated_batch)

        loss_D = -(torch.mean(pred_real) - torch.mean(pred_fake))

        self.opt_D.zero_grad()
        loss_D.backward()
        self.opt_D.step()

        for p in self.D.parameters():
            p.data.clamp_(-self.clip, self.clip)

        if self.step%5 == 0:
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
