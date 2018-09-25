import torch
import torch.nn as nn

from base_network import TinyMultiTaskResNet
from utils import parameter_magnitude, gradient_magnitude, update_magnitude, update_ratio





class _FCDiscriminator(nn.Module):
    """
    FC Discriminator network, which is just a special case of the TinyMultiTaskResNet.
    """
    def __init__(self, input_size=48, hidden_size=256, layers_per_block=2, num_blocks=1):
        super(_FCDiscriminator, self).__init__()
        self.base_network = TinyMultiTaskResNet(input_size, hidden_size, [1], layers_per_block, num_blocks)

    def forward(self, x):
        return self.base_network(x)[0]





class _FCGenerator(nn.Module):
    """
    FC Generator Network, which is also just a special case of the TinyMultiTaskResNet
    """
    def __init__(self, latent_size=10, hidden_size=256, output_size=48, layers_per_block=2, num_blocks=1):
        super(_FCGenerator, self).__init__()
        self.base_network = TinyMultiTaskResNet(latent_size, hidden_size, [output_size], layers_per_block, num_blocks)

    def forward(self, x):
        return self.base_network(x)[0]





class FullyConnectedGan(nn.Module):
    """
    FC Wasserstien Generative Adversarial Network, to generate 1D data. Ties together the logic for training the _FCDiscriminator
    and _FCGenerator. This network implements functions to be able to use the generic training loop in
    utils.training_utils, which requires networks to have a "model_name" attribute.

    The size of the data to be generated is specified with 'data_size' and the dimension of the latent random
    variables used in the generation process is 'latent_size'.

    This is a Wasserstein GAN [https://arxiv.org/abs/1701.07875], and implements parameter clipping, clipped
    to the values [-clip_max, clip_max]. We also implement the gradient penalty from [https://arxiv.org/pdf/1704.00028].
    """
    def __init__(self, data_size=48, latent_size=10, clip_max=0.01):
        super(FullyConnectedGan, self).__init__()
        self.model_name = "fully_connected_gan"

        self.data_size = data_size
        self.latent_size = latent_size
        self.clip_max = clip_max

        self.discr = _FCDiscriminator(input_size=data_size)
        self.gen = _FCGenerator(latent_size=latent_size, output_size=data_size)


    def sample(self, n):
        """
        Use the GAN in genrative mode, sampling latent variables and using the generator.

        :param n: The number of samples desired
        :return: The samples, returned as a batch with shape (n, self.data_size)
        """
        shape = torch.Size((n, self.latent_size))
        latents = torch.autograd.Variable(torch.rand(shape) * 2.0 - 1.0, requires_grad=True).cuda() # uni[-1,1]
        return self.gen(latents)


    def discriminator(self, x):
        return self.discr(x)


    def generator(self, x):
        return self.gen(x)


    def discriminator_params(self):
        return self.discr.parameters()


    def generator_params(self):
        return self.gen.parameters()


    def forward(self, x):
        raise NotImplementedException("Souldn't call GAN directly. Use discriminator(x) or generator(x).")
