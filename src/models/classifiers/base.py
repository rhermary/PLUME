from abc import ABCMeta, abstractmethod
from typing import Any

from torch import nn
import torch

from models.base import Base


class BaseClassifier(nn.Module, Base, metaclass=ABCMeta):
    @abstractmethod
    def forward(  # pylint: disable=arguments-differ
        self, inputs: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Any:
        pass

    @abstractmethod
    def in_channels(self) -> int:
        pass


class Classifier2D(BaseClassifier, metaclass=ABCMeta):
    """Common architecture of 2D classifiers."""

    def __init__(self) -> None:
        """Initialize the network.

        This classifier version is the heavy one used for CIFAR10 in the
        original article. It is based on the code available in the openreview
        files.
        """
        super().__init__()

        self.rep_dim = 64

        # CIFAR10-adapted network params
        self.depth = 4
        self.input_size = 32

        self.convs = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.in_channels()
                        if hidden == 0
                        else 16 * pow(2, hidden - 1),
                        out_channels=16 * pow(2, hidden),
                        kernel_size=5,
                        bias=False,
                        padding=2,
                    ),
                    nn.BatchNorm2d(
                        16 * pow(2, hidden), eps=1e-04, affine=False
                    ),
                    nn.LeakyReLU(),
                    nn.MaxPool2d(2, 2),
                )
                for hidden in range(self.depth)
            ]
        )

        self.fc1 = nn.Linear(
            in_features=pow(self.input_size // pow(2, self.depth), 2)
            * (16 * pow(2, self.depth - 1)),
            out_features=128,
            bias=False,
        )

        self.fc2 = nn.Linear(128, self.rep_dim, bias=False)
        self.fc3 = nn.Linear(self.rep_dim, 1, bias=False)

    def in_channels(self) -> int:
        # CIFAR10-adapted network params
        return 3
    
class Classifier1D(BaseClassifier, metaclass=ABCMeta):
    """Common architecture of 1D classifiers."""

    def __init__(self) -> None:
        super().__init__()

        self.input_size = 32
        self.input_shape = 3 * 32 * 32

        self.fc1 = nn.Sequential(
            nn.Linear(
                in_features=self.input_shape,
                out_features=1024,
                bias=False,
            ),
            nn.BatchNorm1d(1024, eps=1e-04, affine=False),
            nn.LeakyReLU(),
            nn.Linear(
                in_features=1024,
                out_features=512,
                bias=False,
            ),
            nn.BatchNorm1d(512, eps=1e-04, affine=False),
            nn.LeakyReLU(),
        )
        self.fc2 = nn.Linear(
            in_features=512,
            out_features=1,
            bias=False,
        )

    def in_channels(self) -> int:
        return 0