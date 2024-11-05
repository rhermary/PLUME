"""Utility classes and functions for the use of datasets."""

from typing import List, Sequence

import numpy as np
import torch
import torch.nn.functional as F


class OneHot:
    """
    Defining a one-hot encoding transform to be used in
    `torchvision.transforms.Compose`
    """

    def __init__(self, num_classes: int = 10) -> None:
        self.num_classes = num_classes

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return F.one_hot(data, self.num_classes)


def get_target_label_idx(
    labels: torch.Tensor,
    targets: Sequence[int],
) -> List[int]:
    """
    Get the indices of labels that are included in targets.
    :param labels: array of labels
    :param targets: list/tuple of target labels
    :return: list with indices of target labels
    """
    return np.argwhere(np.isin(labels, targets)).flatten().tolist()
