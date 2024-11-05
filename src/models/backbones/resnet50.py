from typing import Tuple, Any, Union, Optional, IO

import torch
from lightning_fabric.utilities.types import _PATH, _MAP_LOCATION_TYPE

from .base import BaseBackbone
from .utils import load_from_checkpoint_
from ..resnet50 import ResNet50


class ResNet50Backbone(BaseBackbone, ResNet50):

    def __init__(
        self,
        load_weights: bool = False,
        num_classes: int = 10,
        **kwargs: Any,
    ) -> None:
        ResNet50.__init__(
            self, load_weights=load_weights, num_classes=num_classes, **kwargs
        )
        self.post_init_()

    def post_init_(self) -> None:
        del self.resnet50.fc

    def forward(
        self, inputs: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        inputs = self.resnet50.conv1(inputs)
        inputs = self.resnet50.bn1(inputs)
        inputs = self.resnet50.relu(inputs)
        inputs = self.resnet50.maxpool(inputs)

        inputs = self.resnet50.layer1(inputs)
        inputs = self.resnet50.layer2(inputs)
        inputs = self.resnet50.layer3(inputs)
        inputs = self.resnet50.layer4(inputs)

        inputs = self.resnet50.avgpool(inputs)
        inputs = torch.flatten(inputs, 1)

        return inputs

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[_PATH, IO],
        map_location: _MAP_LOCATION_TYPE = None,
        hparams_file: Optional[_PATH] = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> "ResNet50Backbone":
        return load_from_checkpoint_(
            cls,
            ResNet50,
            checkpoint_path,
            map_location,
            hparams_file,
            strict,
            **kwargs,
        )
