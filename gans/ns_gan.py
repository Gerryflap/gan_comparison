import torch

from gans.vanilla_gan import VanillaGan


class NonSaturatingGan(VanillaGan):
    """
        The GAN from the original GAN paper with the non-saturating loss
    """
    @staticmethod
    def _generator_loss(pred_fake):
        return -torch.mean(torch.log(pred_fake + 1e-6))
