"""Anomaly classifiers."""

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .base import Classifier2D, BaseClassifier, Classifier1D


class PLADClassifier(Classifier2D):
    """Classifier implementation from PLAD paper based on LeNet."""

    def forward(
        self, inputs: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        r"""Forward method.

        Args:
            inputs (torch.Tensor):
                :math:`(N, in_channels, input_size, input_size)` reshapable
                tensor, where :math:`N` is the batch size.

        Returns:
            torch.Tensor: :math:`(N, 1)`, the classification prediction for each
            batch element.
        """
        inputs = inputs.view(
            inputs.shape[0], self.in_channels(), self.input_size, self.input_size
        )
        inputs = self.convs(inputs)

        inputs = inputs.view(inputs.size(0), -1)

        inputs = F.leaky_relu(self.fc1(inputs))
        inputs = F.leaky_relu(self.fc2(inputs))
        inputs = self.fc3(inputs)

        return inputs


class Classifier1DClassic(Classifier1D):
    def forward(
        self, inputs: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        inputs = inputs.view(inputs.shape[0], self.input_shape)

        return self.fc2(self.fc1(inputs))

# TODO use base classifier
class PLADClassifier1D(BaseClassifier):
    def __init__(self) -> None:
        super().__init__()

        self.rep_dim = 64

        # CIFAR10-adapted network params
        self.depth = 4
        self.input_size = 32

        self.input_shape = self.in_channels() * self.input_size**2

        # https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/?rdt=38176
        # https://forums.fast.ai/t/why-perform-batch-norm-before-relu-and-not-after/81293/2
        # https://github.com/keras-team/keras/issues/1802#issuecomment-187966878
        # https://github.com/gcr/torch-residual-networks/issues/5
        self.fcs = nn.Sequential(
            nn.BatchNorm1d(self.input_shape, eps=1e-04, affine=False),  # TODO
            nn.Linear(
                in_features=self.input_shape,
                out_features=110,
                bias=False,
            ),
            nn.BatchNorm1d(110, eps=1e-04, affine=False),
            nn.LeakyReLU(),
            nn.Linear(
                in_features=110,
                out_features=53,
                bias=False,
            ),
            nn.BatchNorm1d(53, eps=1e-04, affine=False),
            nn.LeakyReLU(),
            nn.Linear(
                in_features=53,
                out_features=1,
                bias=False,
            ),
        )

    def forward(
        self, inputs: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        inputs = inputs.view(inputs.shape[0], self.input_shape)

        return self.fcs(inputs)
    
    def in_channels(self) -> int:
        return 3


class PLADClassifier1DSimple(BaseClassifier):
    def __init__(self) -> None:
        super().__init__()

        self.rep_dim = 64

        # CIFAR10-adapted network params
        self.depth = 4
        self.input_size = 32

        self.input_shape = self.in_channels() * self.input_size**2

        self.fcs = nn.Sequential(
            nn.BatchNorm1d(self.input_shape, eps=1e-04, affine=False),
            nn.Linear(
                in_features=self.input_shape,
                out_features=200,
                bias=False,
            ),
            nn.BatchNorm1d(200, eps=1e-04, affine=False),
            nn.LeakyReLU(),
            nn.Linear(
                in_features=200,
                out_features=1,
                bias=False,
            ),
        )

    def forward(
        self, inputs: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        inputs = inputs.view(inputs.shape[0], self.input_shape)

        return self.fcs(inputs)

    def in_channels(self) -> int:
        return 3