"""Combining feature extractor and anomaly detection framework."""

from typing import Union, Dict, Any, Tuple, Optional
import itertools

from torch import nn
import torch

from .plad import PLAD
from .backbones.base import BaseBackbone
from models.base import BaseAdaptativeFeatureExtractor


class FeaturePLAD(PLAD):
    """Extension of PLAD for the network to work with a backbone.

    Essentially, the anomaly detection part will be on features instead of
    directly on images.
    """

    def __init__(
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

        extractor_output_dim = 10
        # Uncomment if using legacy code
        # extractor_output_dim = (
        #     self.feature_extractor.feature_total
        #     * self.feature_extractor.num_classes
        # )

        self.features_given_directly = features_given_directly
        self.dense = nn.Linear(extractor_output_dim, 3072)
        self.batch_norm = nn.BatchNorm1d(3072, affine=False)

    @classmethod
    def input_shape(cls) -> Tuple[int, int, int]:
        raise RuntimeError("The input shape depends on the backbone class")

    def configure_optimizers(
        self,
    ) -> Union[torch.optim.Optimizer, Dict[str, Any]]:
        return self.configure_optimizers_(
            itertools.chain(
                self.classifier.parameters(),
                self.vae.parameters(),
                self.dense.parameters(),
            ),
        )

    def forward(
        self, inputs: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Tuple[torch.Tensor, ...]:
        features = self._feature_extraction(inputs, *args, **kwargs)
        return PLAD.forward(self, features.view(-1, *PLAD.input_shape()))

    def _feature_extraction(
        self, inputs: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """
        The part of the forward pass that changes between this class and its
        subclass `FeaturePladReduced` which does not need the addition of a
        dense layer.
        """
        if self.features_given_directly:
            features = inputs.reshape(inputs.shape[0], -1)
        else:
            features = self.feature_extractor(inputs)

        features = self.dense(features.reshape(features.shape[0], -1))
        features = self.batch_norm(features)

        return features

    def validation_step(  # pylint: disable=arguments-differ
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        inputs, targets = batch
        features = self._feature_extraction(inputs)

        return PLAD.validation_step(self, (features, targets), *args, **kwargs)


class FeaturePLADReduced(FeaturePLAD):
    """
    PLAD working on features but with the dense layer to adapt the features
    dimensions to PLAD directly in the backbone.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        classifier: nn.Module,
        vae: nn.Module,
        lambda_: float,
        backbone: BaseBackbone,
        features_given_directly: bool = False,
        optimizer: Optional[
            Union[torch.optim.Optimizer, Dict[str, Any]]
        ] = None,
        require_train_stat: bool = True,
        epochs: int = 100,
        lr: float = 0.005,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            classifier=classifier,
            vae=vae,
            lambda_=lambda_,
            optimizer=optimizer,
            require_train_stat=require_train_stat,
            epochs=epochs,
            lr=lr,
            backbone=backbone,
            features_given_directly=features_given_directly,
            params=kwargs.pop("params", locals()),
            **kwargs,
        )

        del self.dense

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
        if self.features_given_directly:
            features = inputs.reshape(inputs.shape[0], -1)
        else:
            _, features = self.feature_extractor(inputs)

        features = self.batch_norm(features)

        return features


class AdaptativeFeaturePLAD(PLAD, BaseAdaptativeFeatureExtractor):
    def __init__(
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
        self.features_given_directly = features_given_directly

        self.adaptor = nn.AdaptiveAvgPool1d(3072)
        self.batch_norm = nn.BatchNorm1d(3072, affine=False)

    @classmethod
    def input_shape(cls) -> Tuple[int, int, int]:
        raise RuntimeError("The input shape depends on the backbone class")

    def configure_optimizers(
        self,
    ) -> Union[torch.optim.Optimizer, Dict[str, Any]]:
        return self.configure_optimizers_(
            itertools.chain(
                self.classifier.parameters(),
                self.vae.parameters(),
                self.adaptor.parameters(),
            ),
        )

    def forward(
        self, inputs: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Tuple[torch.Tensor, ...]:
        features = self._feature_extraction(inputs, *args, **kwargs)
        return PLAD.forward(self, features.view(-1, *PLAD.input_shape()))

    def _feature_extraction(
        self, inputs: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """
        The part of the forward pass that changes between this class and its
        subclass `FeaturePladReduced` which does not need the addition of a
        dense layer.
        """
        if self.features_given_directly:
            features = inputs.reshape(inputs.shape[0], -1)
        else:
            features = self.feature_extractor(inputs)

        features = self.adaptor(features.reshape(features.shape[0], -1))
        features = self.batch_norm(features)

        return features

    def validation_step(  # pylint: disable=arguments-differ
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        inputs, targets = batch
        features = self._feature_extraction(inputs)

        return PLAD.validation_step(self, (features, targets), *args, **kwargs)

