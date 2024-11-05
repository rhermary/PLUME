from typing import Tuple, Any, Union, Optional, IO

import torch
from lightning_fabric.utilities.types import _PATH, _MAP_LOCATION_TYPE

from .base import BaseBackbone
from .utils import load_from_checkpoint_
from ..vgg16 import VGG16


class VGG16Backbone(BaseBackbone, VGG16):

    def __init__(
        self,
        load_weights: bool = False,
        num_classes: int = 10,
        **kwargs: Any,
    ) -> None:
        VGG16.__init__(
            self, load_weights=load_weights, num_classes=num_classes, **kwargs
        )
        self.post_init_()

    def post_init_(self) -> None:
        self.vgg.classifier._modules.pop('6')

    def forward(
        self, inputs: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        return self.vgg(inputs)

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[_PATH, IO],
        map_location: _MAP_LOCATION_TYPE = None,
        hparams_file: Optional[_PATH] = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> "VGG16Backbone":
        return load_from_checkpoint_(
            cls,
            VGG16,
            checkpoint_path,
            map_location,
            hparams_file,
            strict,
            **kwargs,
        )
