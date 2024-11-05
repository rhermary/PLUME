"""Combining feature extractor and anomaly detection framework."""

from typing import Union, Dict, Any, Tuple, Optional
import itertools

from torch import nn
import torch

from .rcsplad import RotCosSimPLAD
from models.backbones.base import BaseBackbone
from models.base import BaseAdaptativeFeatureExtractor


class FeatureRCSPLAD(RotCosSimPLAD):
    def __init__(  # pylint: disable=too-many-arguments
        self,
        backbone: BaseBackbone,
        features_given_directly: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            params=kwargs.pop("params", locals()),
            **kwargs,
        )

        self.feature_extractor = backbone
        if features_given_directly:
            self.feature_extractor = None

        self.batch_norm = nn.BatchNorm1d(3072, affine=False)

    @classmethod
    def input_shape(cls) -> Tuple[int, int, int]:
        raise RuntimeError("The input shape depends on the backbone class")

    def forward(
        self, inputs: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Tuple[torch.Tensor, ...]:
        features = self._feature_extraction(inputs, *args, **kwargs)
        return RotCosSimPLAD.forward(
            self, features.view(-1, *RotCosSimPLAD.input_shape())
        )

    def validation_step(  # pylint: disable=arguments-differ
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        inputs, targets = batch
        features = self._feature_extraction(inputs)

        return RotCosSimPLAD.validation_step(
            self, (features, targets), *args, **kwargs
        )

    def configure_optimizers(
        self,
    ) -> Union[torch.optim.Optimizer, Dict[str, Any]]:
        return self.configure_optimizers_(
            itertools.chain(
                self.classifier.parameters(),
                self.vae.parameters(),
            ),
        )

    def _feature_extraction(
        self, inputs: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        if self.feature_extractor is None:
            features = inputs.reshape(inputs.shape[0], -1)
        else:
            _, features = self.feature_extractor(inputs)

        features = self.batch_norm(features)

        return features


class AdaptativeFeatureRCSPLAD(FeatureRCSPLAD, BaseAdaptativeFeatureExtractor):
    def __init__(  # pylint: disable=too-many-arguments
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            params=kwargs.pop("params", locals()),
            **kwargs,
        )

        self.adaptor = nn.AdaptiveAvgPool1d(3072)

    def _feature_extraction(
        self, inputs: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        if self.feature_extractor is None:
            features = inputs.reshape(inputs.shape[0], -1)
        else:
            features = self.feature_extractor(inputs)

        features = self.adaptor(features)
        features = self.batch_norm(features)

        return features
