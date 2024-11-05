"""VAE implementation and declinations."""

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


class VAE(nn.Module):
    """VAE implementation of PLAD paper."""

    def __init__(
        self, input_dim: int = 3072, h_dim: int = 3072, z_dim: int = 3072
    ) -> None:
        """Initialize the VAE.

        Args:
            input_dim (int, optional): 1D (e.g. flattened 2D) size of the input.
            Defaults to 784.
            h_dim (int, optional): size of the hidden layer. Defaults to 784.
            z_dim (int, optional): dimension of the mean and std vectors
            produced by the decoder. Defaults to 784.
        """
        super().__init__()
        # encoder part
        self.fc1 = nn.Linear(input_dim, h_dim)

        self.fc2 = nn.Linear(h_dim, z_dim)  # mu

        self.fc3 = nn.Linear(h_dim, z_dim)  # log_sigma

        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim)

        self.fc5 = nn.Linear(h_dim, input_dim * 2)

    def forward(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward operation of the VAE.

        Args:
            inputs (torch.Tensor): input tensor of the shape
            :math:`(N, input_dim)`

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: in first position,
            the output of the VAE decoder, `[additive:multiplicative]` noise of
            shape :math:`(N, 2 * input_dim)` (their order depends on their use
            afterwards). In second position, the encoded mean of the latent
            distribution, and in third the :math:`log` of its squared variance.
            Both are of shape :math:`(N, z_dim)`.
        """
        mean, logvar = self.encode(inputs)
        sampled_z = self.reparametrize(mean, logvar)
        res = self.decode(sampled_z)

        return res, mean, logvar

    def encode(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encoding part of the VAE.

        Args:
            inputs (torch.Tensor): input tensor of the shape
            :math:`(N, input_dim)`

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: in first position, the encoded
            mean of the latent distribution, and in second the :math:`log` of
            its squared variance. Both are of shape :math:`(N, z_dim)`.
        """
        features = F.leaky_relu(self.fc1(inputs))
        mean = self.fc2(features)
        logvar = self.fc3(features)  # estimate log(sigma**2) actually
        return mean, logvar

    def reparametrize(
        self, mean: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Given a standard gaussian distribution epsilon ~ N(0,1),
        we can sample the random variable z as per z = mu + sigma * epsilon

        Args:
            mean (torch.Tensor): mean of the latent distribution,
            :math:`(N, z_dim)`
            logvar (torch.Tensor): std of the latent distribution,
            :math:`(N, z_dim)`

        Returns:
            torch.Tensor: sampled element from the latent distribution,
            :math:`(N, z_dim)`
        """
        std = torch.exp(logvar * 0.5)
        eps = torch.randn_like(std)
        return mean + std * eps

    def decode(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Given a sampled z, decode it back to "image" (actually decode noise)

        Args:
            samples (torch.Tensor): sampled vectors from latent distribution
            :math:`(N, z_dim)`

        Returns:
            torch.Tensor: :math:`(N, 2 * input_dim)`
        """
        features = F.leaky_relu(self.fc4(samples))
        res = self.fc5(features)
        return res
