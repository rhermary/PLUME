"""Defines a new model to easily bypass original PLAD functions."""

import types
from typing import Union, Tuple, Any

import torch
from torch.nn import functional as F

from models import PLAD
from models.classifiers.classifier import PLADClassifier


class PLADInspector(PLAD):
    """By-passer for the PLAD model."""

    class PLADClassifierInspector(PLADClassifier):
        use_batch_norm: bool

        def forward_(self, batch: torch.Tensor) -> torch.Tensor:
            batch = batch.view(
                batch.shape[0],
                self.in_channels,
                self.input_size,
                self.input_size,
            )
            if self.use_batch_norm:
                batch = self.convs(self.batch_norm(batch))
            else:
                batch = self.convs(batch)

            batch = batch.view(batch.size(0), -1)

            return F.leaky_relu(self.fc1(batch))

    classifier: PLADClassifierInspector
    predict_pseudo_anomalies: bool
    return_pseudo_anomalies: bool
    truncated: bool

    class TruncatedClassifier(PLADClassifierInspector):
        """Bypass last fully connected layer"""

        def forward(
            self, inputs: torch.Tensor, *args: Any, **kwargs: Any
        ) -> torch.Tensor:
            inputs = self.forward_(inputs)
            return self.fc2(inputs)

    class NoBatchNormClassifier(PLADClassifierInspector):
        def forward(
            self, inputs: torch.Tensor, *args: Any, **kwargs: Any
        ) -> torch.Tensor:
            inputs = self.forward_(inputs)
            inputs = F.leaky_relu(self.fc2(inputs))
            inputs = self.fc3(inputs)

            return inputs

    def truncate(self) -> None:
        """
        Permanently Modifies the model to return the before last layer features
        instead of classification results.
        """
        self.truncated = True

        del self.classifier.fc3
        self.classifier.forward = types.MethodType(  # type: ignore
            self.TruncatedClassifier.forward, self.classifier
        )

    def on_pseudo_anomalies(self) -> None:
        """
        Activates the generation of noised images and classify them instead of
        the normal images.
        Reset after validation ends.
        """
        self.predict_pseudo_anomalies = True

    def get_pseudo_anomalies(self) -> None:
        """
        Activates the generation of noised images and directly returns them.
        Reset after validation ends.
        """
        self.predict_pseudo_anomalies = True
        self.return_pseudo_anomalies = True

    def deactivate_batch_norm(self) -> None:
        self.classifier.use_batch_norm = False

    def predict_step(  # pylint: disable=arguments-differ
        self,
        batch: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        if not isinstance(batch, torch.Tensor):
            # If targets are given, unpack the inputs
            batch, _ = batch

        if self.predict_pseudo_anomalies:
            batch_shape = batch.shape
            batch = batch.view(batch_shape[0], -1)
            entity_size = batch.shape[1]

            noise, _, _ = self.vae(batch)
            add_noise, mult_noise = torch.split(noise, entity_size, dim=1)

            batch = batch.view(batch_shape) * mult_noise.view(
                batch_shape
            ) + add_noise.view(batch_shape)

        if self.return_pseudo_anomalies:
            return batch

        return torch.squeeze(self.classifier(batch), dim=1)

    def on_predict_start(self) -> None:
        """
        Avoid overriding `__init__()` and thus checkpoint loading discrepancies.
        """
        if not hasattr(self, "predict_pseudo_anomalies"):
            self.predict_pseudo_anomalies = False

        if not hasattr(self, "return_pseudo_anomalies"):
            self.return_pseudo_anomalies = False

        if not hasattr(self.classifier, "use_batch_norm"):
            self.classifier.use_batch_norm = True

        if not hasattr(self, "truncated"):
            self.truncated = False
            self.classifier.forward = types.MethodType(  # type: ignore
                self.NoBatchNormClassifier.forward, self.classifier
            )

        self.classifier.forward_ = types.MethodType(  # type: ignore
            self.PLADClassifierInspector.forward_, self.classifier
        )

    def on_predict_end(self) -> None:
        self.predict_pseudo_anomalies = False
        self.return_pseudo_anomalies = False
        self.classifier.use_batch_norm = True
