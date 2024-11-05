from typing import Any, Tuple
from torch import Tensor
import torch

from models.classifiers.leaky_classifier import LeakyClassifier
from .csplad import CosSimPLAD
from misc.losses import pseudo_loss


class RotCosSimPLAD(CosSimPLAD):
    def forward(
            self, inputs: Tensor, *args: Any, **kwargs: Any
    ) -> Tuple[Tensor, ...]:
        batch_shape = inputs.size()
        batch_1d = inputs.view(batch_shape[0], -1)
        entity_size = batch_1d.size()[1]

        noise, z_mean, z_logvar = self.vae(batch_1d)
        
        identity_matrix = torch.eye(n=batch_1d.shape[1], device=inputs.device)
        alpha, beta = torch.split(noise, entity_size, dim=1)
        alpha, beta = alpha[:, :, None], beta[:, :, None]
        
        pseudo_rotation_matrix = (
            identity_matrix + torch.bmm(alpha, torch.transpose(beta, 1, 2))
        )

        anomalies = torch.bmm(pseudo_rotation_matrix, batch_1d[:, :, None])
        anomalies = anomalies.reshape(batch_shape)

        self.constraint_loss_ = pseudo_loss(batch_1d, anomalies.view(batch_1d.shape), self.cosine_similarity_)

        classif_normal, features_normal = self.classifier(inputs)
        classif_anomalies, features_anomalies = self.classifier(anomalies)

        classif_normal = torch.squeeze(classif_normal, dim=1)
        classif_anomalies = torch.squeeze(classif_anomalies, dim=1)

        return (
            alpha,
            beta,
            z_mean,
            z_logvar,
            classif_normal,
            classif_anomalies,
            features_normal,
            features_anomalies,
        )