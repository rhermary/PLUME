"""The modified architectures of models when used as backbones."""

from .vgg16 import VGG16Backbone
from .resnet50 import ResNet50Backbone
from .convnext import ConvNeXtBackbone

from .base import BaseBackbone

BACKBONES = BaseBackbone.listing()
