"""PLAD classifier adapted for cosine similarity loss measure in feature space."""

from typing import Any, Tuple
from abc import ABCMeta, abstractmethod

import torch
import torch.nn.functional as F

from .base import Classifier2D, Classifier1D


class LeakyClassifier(Classifier2D, metaclass=ABCMeta):
    @abstractmethod
    def forward(
        self, inputs: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class LeakyClassifier1D(Classifier1D, metaclass=ABCMeta):
    @abstractmethod
    def forward(
        self, inputs: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class DecPLADClassifier(LeakyClassifier):
    """PLAD LeNet-based classifier for modified loss.

    The forward method of this classifier also returns the features of the
    before-last layer (before the actual classification layer, and before
    leaky RELU), so that a Cosine Similarity metric can be computed in feature
    space and added to the loss.
    """

    def forward(
        self, inputs: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward method.

        Args:
            inputs (torch.Tensor):
                :math:`(N, in_channels, input_size, input_size)` reshapable
                tensor, where :math:`N` is the batch size.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: :math:`((N, 1), (N, 64))`, the
            classification prediction for each batch element, and the
            before-last layer features.
        """
        inputs = inputs.view(
            inputs.shape[0], self.in_channels(), self.input_size, self.input_size
        )
        inputs = self.convs(inputs)

        inputs = inputs.view(inputs.size(0), -1)

        inputs = F.leaky_relu(self.fc1(inputs))
        features = self.fc2(inputs)

        inputs = self.fc3(F.leaky_relu(features))

        return inputs, features

class DecClassifier1D(LeakyClassifier1D):
    def forward(
        self, inputs: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = inputs.view(inputs.shape[0], self.input_shape)

        features = self.fc1(inputs)
        inputs = self.fc2(features)

        return inputs, features

class InfPLADClassifier(LeakyClassifier):
    """PLAD LeNet-based classifier for modified loss.

    The forward method of this classifier also returns the features extracted by
    the last convolutional layer, so that a Cosine Similarity metric can be
    computed in feature space and added to the loss.
    """

    def forward(
        self, inputs: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward method.

        Args:
            inputs (torch.Tensor):
                :math:`(N, in_channels, input_size, input_size)` reshapable
                tensor, where :math:`N` is the batch size.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: :math:`((N, 1), (N, 64))`, the
            classification prediction for each batch element, and the
            last convolutional layer features.
        """
        inputs = inputs.view(
            inputs.shape[0], self.in_channels(), self.input_size, self.input_size
        )
        inputs = self.convs(inputs)

        features = inputs.view(inputs.size(0), -1)

        inputs = F.leaky_relu(self.fc1(features))
        inputs = F.leaky_relu(self.fc2(inputs))
        inputs = self.fc3(inputs)

        return inputs, features
