from typing import Any, Tuple, Union

from torch import Tensor

from models.backbones import VGG16Backbone


class VGG16Inspector(VGG16Backbone):
    def predict_step(  # pylint: disable=arguments-differ
        self,
        batch: Union[Tuple[Tensor, Tensor], Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        if not isinstance(batch, Tensor):
            # If targets are given, unpack the inputs
            batch, _ = batch

        output = self(batch)
        return output
