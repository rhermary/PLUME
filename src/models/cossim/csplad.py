"""PLAD network adapted to CosSimClassifier and cosine similarity loss."""

from typing import Dict, Tuple, Any, Optional

import torch
from torch import nn
from torch.optim import Optimizer
import torch.nn.functional as F

from models.plad import (
    PLADBase,
    VAL_LOSS,
)
from models.classifiers.leaky_classifier import LeakyClassifier
from misc.losses import (
    kl_divergence,
    noise_loss,
    classif_loss,
    cossim_coercion_loss,
    cossim_constraint_loss,
    contrastive_loss,
    mean_contrastive_loss,
    pseudo_loss,
)

COSSIM_CONS_LOSS = "cossim_cons_loss"
COSSIM_COER_LOSS = "cossim_coer_loss"
CONTRASTIVE_LOSS = "contrastive_loss"
MEAN_CONTRASTIVE_LOSS = "mean_contrastive_loss"


class CosSimPLAD(PLADBase):
    """Cosine Similarity augmented PLAD.

    In addition to all PLAD features, this network take the features extracted
    by the classifier to calculate a cosine similarity between the normal data
    and the pseudo-anomalies; this metric is wanted to be maximized during the
    training, and hence added to the loss accordingly.
    """

    def __init__(
        self,
        classifier: LeakyClassifier,
        eta: float,
        nu: float,
        gamma: float,
        phi: float,
        tau: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            classifier,
            params=kwargs.pop("params", locals()),
            **kwargs,
        )

        self.cosine_similarity_ = torch.nn.CosineSimilarity(dim=2, eps=1e-08)
        self.eta = eta
        self.nu = nu
        self.gamma = gamma
        self.phi = phi
        self.tau = tau

    def forward(  # pylint: disable=arguments-differ,too-many-locals
        self, inputs: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass.

        Args:
            inputs (torch.Tensor): (N, VAE_in) reshapable tensor.

        Returns:
            Tuple[ torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                additive noise, multiplicative noise, embedded mean, embedded
                variance, classification results of the normal samples,
                classification results of the pseudo-anomalies, features
                extracted on normal data and pseudo-anomalies, respectively.
        """
        batch_shape = inputs.size()
        batch_1d = inputs.view(batch_shape[0], -1)
        entity_size = batch_1d.size()[1]

        noise, z_mean, z_logvar = self.vae(batch_1d)
        add_noise, mult_noise = torch.split(noise, entity_size, dim=1)

        anomalies = inputs * mult_noise.view(batch_shape) + add_noise.view(
            batch_shape
        )
        
        self.constraint_loss_ = pseudo_loss(batch_1d, anomalies.view(batch_1d.shape), self.cosine_similarity_)

        classif_normal, features_normal = self.classifier(inputs)
        classif_anomalies, features_anomalies = self.classifier(anomalies)

        classif_normal = torch.squeeze(classif_normal, dim=1)
        classif_anomalies = torch.squeeze(classif_anomalies, dim=1)

        return (
            add_noise,
            mult_noise,
            z_mean,
            z_logvar,
            classif_normal,
            classif_anomalies,
            features_normal,
            features_anomalies,
        )

    def _loss(
        self,
        noise_loss_: torch.Tensor,
        classif_loss_: torch.Tensor,
        kl_divergence_: torch.Tensor,
        cossim_constraint_loss_: torch.Tensor,
        cossim_coercion_loss_: torch.Tensor,
        contrastive_loss_: torch.Tensor,
        mean_contrastive_loss_: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Sums the 3 base losses and the cosine similarity loss."""

        return (
            classif_loss_
            + kl_divergence_
            + self.lambda_ * noise_loss_
            + self.eta * cossim_constraint_loss_
            + self.nu * cossim_coercion_loss_
            + self.gamma * contrastive_loss_
            + self.phi * mean_contrastive_loss_
        )

    def training_step(  # pylint: disable=arguments-differ
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        # pylint: disable=too-many-locals
        inputs, targets = batch

        (
            add_noise,
            mult_noise,
            z_mean,
            z_logvar,
            classif_normal,
            classif_anomalies,
            normal_features,
            anomalies_features,
        ) = self(inputs)

        noise_loss_ = noise_loss(add_noise, mult_noise)
        classif_loss_ = classif_loss(
            classif_normal, classif_anomalies, torch.squeeze(targets)
        )
        kl_divergence_ = kl_divergence(z_mean, z_logvar)

        cossim_coercion_loss_ = cossim_coercion_loss(
            normal_features, self.cosine_similarity_
        )
        contrastive_loss_ = contrastive_loss(
            normal_features, anomalies_features, self.cosine_similarity_, self.tau
        )
        mean_contrastive_loss_ = mean_contrastive_loss(
            normal_features, anomalies_features, self.cosine_similarity_, self.tau
        )

        cossim_constraint_loss_ = self.constraint_loss_

        loss = self._loss(
            noise_loss_,
            classif_loss_,
            kl_divergence_,
            cossim_constraint_loss_,
            cossim_coercion_loss_,
            contrastive_loss_,
            mean_contrastive_loss_,
        )

        self.log_train_state(
            loss=loss,
            classif_loss_=classif_loss_,
            kl_divergence_=kl_divergence_,
            noise_loss_=noise_loss_,
            **{
                COSSIM_CONS_LOSS: cossim_constraint_loss_,
                COSSIM_COER_LOSS: cossim_coercion_loss_,
                CONTRASTIVE_LOSS: contrastive_loss_,
                MEAN_CONTRASTIVE_LOSS: mean_contrastive_loss_,
            },
        )

        return loss

    def validation_step(  # pylint: disable=arguments-differ
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        inputs, targets = batch
        classif_results, _ = self.classifier(inputs)
        classif_results = torch.squeeze(classif_results, dim=1)

        loss = F.binary_cross_entropy_with_logits(classif_results, targets)

        self.log(VAL_LOSS, loss, sync_dist=True)

        return {"loss": loss, "logits": classif_results}
