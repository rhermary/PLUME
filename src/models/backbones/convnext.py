from typing import Tuple, Any, Union, Optional, IO

import torch
from lightning_fabric.utilities.types import _PATH, _MAP_LOCATION_TYPE

from .base import BaseBackbone
from .utils import load_from_checkpoint_
from ..convnext import ConvNeXt


class ConvNeXtBackbone(BaseBackbone, ConvNeXt):

    def __init__(
        self,
        load_weights: bool = False,
        num_classes: int = 10,
        **kwargs: Any,
    ) -> None:
        ConvNeXt.__init__(
            self, load_weights=load_weights, num_classes=num_classes, **kwargs
        )
        self.post_init_()

    def post_init_(self) -> None:
        del self.convnext.classifier

    def forward(
        self, inputs: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        inputs = self.convnext.features(inputs)
        inputs = self.convnext.avgpool(inputs)

        return inputs

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[_PATH, IO],
        map_location: _MAP_LOCATION_TYPE = None,
        hparams_file: Optional[_PATH] = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> "ConvNeXtBackbone":
        return load_from_checkpoint_(
            cls,
            ConvNeXt,
            checkpoint_path,
            map_location,
            hparams_file,
            strict,
            **kwargs,
        )
