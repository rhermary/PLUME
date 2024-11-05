"""Threshold calculated based on the training dataset results."""

from typing import Any
import math

from torchmetrics import Metric
import torch


class DynamicThreshold(Metric):
    """ "Defines an extreme value theory-based threshold."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.percentile = 0.1

        self.add_state("outputs", torch.tensor([]), dist_reduce_fx=None)
        self.outputs: torch.Tensor

    def update(  # pylint: disable=arguments-differ
        self, predictions: torch.Tensor, logits: bool = True
    ) -> None:
        if logits:
            predictions = torch.sigmoid(predictions)

        self.outputs = torch.cat((self.outputs, predictions))

    def compute(self) -> float:
        kth, _ = torch.kthvalue(
            self.outputs.flatten(),
            math.ceil(self.outputs.shape[0] * self.percentile),
        )

        return kth.item()

    @property
    def computed(self) -> bool:
        """True if the threshold was computed for this epoch."""
        return not self._computed is None
