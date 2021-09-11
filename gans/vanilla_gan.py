import torch

from gans.abstract_gan import AbstractGan


class VanillaGan(AbstractGan):
    """
        The GAN from the original GAN paper without the non-saturating loss
    """
    def _train_step(self, real_batch):
        # Train D
        generated_batch = self.generate_batch(real_batch.size(0)).detach()

        pred_real = torch.sigmoid(self.D(real_batch))
        pred_fake = torch.sigmoid(self.D(generated_batch))

        loss_D = -torch.mean(torch.log(pred_real + 1e-6) + (torch.log(1.0 - pred_fake + 1e-6)))

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

    @staticmethod
    def _generator_loss(pred_fake):
        # Saturating loss
        return torch.mean(torch.log(1.0 - pred_fake + 1e-6))
