"""Defines a new model to easily bypass original PLAD functions."""

import types
from typing import Union, Tuple, Any

import torch
from torch.nn import functional as F

from models.cossim.rcsplad import RotCosSimPLAD
from models import DecClassifier1D


class RCSPLADInspector(RotCosSimPLAD):
    class TruncatedClassifier(DecClassifier1D):
        def forward(
            self, inputs: torch.Tensor, *args: Any, **kwargs: Any
        ) -> torch.Tensor:
            inputs = inputs.view(inputs.shape[0], self.input_shape)

            features = self.fc2(self.fc1(inputs))

            return features
        
        
    class OldVersionClassifier(DecClassifier1D):
        def forward(
            self, inputs: torch.Tensor, *args: Any, **kwargs: Any
        ) -> torch.Tensor:
            inputs = inputs.view(inputs.shape[0], self.input_shape)

            features = self.fc2(self.fc1(inputs))

            return features
        

    classifier: TruncatedClassifier
    predict_pseudo_anomalies: bool
    return_pseudo_anomalies: bool
    return_alpha_beta: bool
    truncated: bool

    def truncate(self) -> None:
        """
        Permanently Modifies the model to return the before last layer features
        instead of classification results.
        """
        self.truncated = True

        del self.classifier.fc2
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

    def get_alpha_beta(self) -> None:
        self.return_alpha_beta = True

    def deactivate_batch_norm(self) -> None:
        pass

    def predict_step(  # pylint: disable=arguments-differ
        self,
        batch: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        if not isinstance(batch, torch.Tensor):
            # If targets are given, unpack the inputs
            batch, _ = batch

        if self.predict_pseudo_anomalies or self.return_alpha_beta:
            batch_shape = batch.size()
            batch_1d = batch.view(batch_shape[0], -1)
            entity_size = batch_1d.size()[1]

            noise, _, _ = self.vae(batch_1d)
            
            identity_matrix = torch.eye(n=batch_1d.shape[1], device=batch.device)
            alpha, beta = torch.split(noise, entity_size, dim=1)
            
            if self.return_alpha_beta:
                return torch.cat((alpha[:, None, :], beta[:, None, :]), dim=1)
            
            alpha, beta = alpha[:, :, None], beta[:, :, None]
            pseudo_rotation_matrix = (
                identity_matrix + torch.bmm(alpha, torch.transpose(beta, 1, 2))
            )

            anomalies = torch.bmm(pseudo_rotation_matrix, batch_1d[:, :, None])
            batch = anomalies.reshape(batch_shape)

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

        if not hasattr(self, "truncated"):
            self.truncated = False
            
        if not hasattr(self, "return_alpha_beta"):
            self.return_alpha_beta = False

        if not hasattr(self.classifier, "fc3"):
            self.classifier.forward = types.MethodType(
                self.OldVersionClassifier.forward, self.classifier
            )
    
    def on_predict_end(self) -> None:
        self.predict_pseudo_anomalies = False
        self.return_pseudo_anomalies = False
        self.return_alpha_beta = False
