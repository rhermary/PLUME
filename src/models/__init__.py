"""Modules for all the implemented models."""

from .base import BaseFeatureExtractor, BaseAdaptativeFeatureExtractor
from .classifiers.base import BaseClassifier
from .classifiers.classifier import PLADClassifier, PLADClassifier1D
from .classifiers.leaky_classifier import DecPLADClassifier, InfPLADClassifier, DecClassifier1D
from .combined import FeaturePLAD, FeaturePLADReduced, AdaptativeFeaturePLAD
from .cossim.cscombined import FeatureCSPLAD, AdaptativeFeatureCSPLAD
from .cossim.rcscombined import FeatureRCSPLAD, AdaptativeFeatureRCSPLAD
from .plad import PLAD
from .vae import VAE
from .cossim.csplad import CosSimPLAD
from .cossim.rcsplad import RotCosSimPLAD
from .vgg16 import VGG16
from .resnet50 import ResNet50
from .convnext import ConvNeXt

from .backbones import BACKBONES

# Even if the backbones are subclasses of BaseFeatureExtractor, they are not
# imported here so they will not appear in the values
FEATURE_EXTRACTORS = BaseFeatureExtractor.listing()
CLASSIFIERS = BaseClassifier.listing()

__all__ = [
    "PLADClassifier",
    "PLADClassifier1D",
    "ConvNeXt",
    "DecPLADClassifier",
    "InfPLADClassifier",
    "FeaturePLAD",
    "FeaturePLADReduced",
    "AdaptativeFeaturePLAD",
    "PLAD",
    "VAE",
    "FEATURE_EXTRACTORS",
    "BACKBONES",
    "CLASSIFIERS",
    "CosSimPLAD",
    "VGG16",
    "ResNet50",
    "RotCosSimPLAD",
    "DecClassifier1D",
    "FeatureCSPLAD",
    "AdaptativeFeatureCSPLAD",
    "BaseAdaptativeFeatureExtractor",
    "FeatureRCSPLAD",
    "AdaptativeFeatureRCSPLAD",
]
