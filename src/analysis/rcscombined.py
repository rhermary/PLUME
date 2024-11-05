from typing import Union, Tuple, Any

import torch

from models.cossim.rcscombined import AdaptativeFeatureRCSPLAD
from .rcsplad import RCSPLADInspector


class FeatureRCSPLADInspector(AdaptativeFeatureRCSPLAD, RCSPLADInspector):
    features_given_directly: bool

    def on_images(self) -> None:
        self.features_given_directly = False

    def predict_step(  # pylint: disable=arguments-differ
        self,
        batch: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Works only with extracted feature dataset"""
        if not isinstance(batch, torch.Tensor):
            # If targets are given, unpack the inputs
            batch, _ = batch

        if not self.features_given_directly:
            batch = self.feature_extractor(batch)

        batch = self.batch_norm(self.adaptor(batch.reshape(batch.shape[0], -1)))

        return super().predict_step(batch, *args, **kwargs)

    def on_predict_start(self) -> None:
        super().on_predict_start()
        if not hasattr(self, "features_given_directly"):
            self.features_given_directly = True

    def on_predict_end(self) -> None:
        super().on_predict_end()
        self.features_given_directly = True
