"""PLAD network implementation."""

from typing import Dict, Any, Optional, Tuple, Union, Iterable
from abc import ABCMeta
import itertools

import torch.nn.functional as F
from torch import nn
import torch

from misc.losses import kl_divergence, noise_loss, classif_loss
from .base import BaseModule
from .classifiers.base import BaseClassifier
from .classifiers.classifier import PLADClassifier

VAL_LOSS = "val_loss"

NOISE_LOSS = "noise_loss"
CLASSIF_LOSS = "classif_loss"
KL_DIVERGENCE = "kl_divergence"
TRAIN_LOSS = "total_train_loss"
LEARNING_RATE = "lr"


class PLADBase(BaseModule, metaclass=ABCMeta):
    """PLAD original implementation as a `LightningModule`"""

    def __init__(  # pylint: disable=invalid-name,too-many-arguments
        self,
        classifier: BaseClassifier,
        vae: nn.Module,
        lambda_: float,
        optimizer: Optional[
            Union[torch.optim.Optimizer, Dict[str, Any]]
        ] = None,
        require_train_stat: bool = False,
        epochs: int = 100,
        steps_per_epoch: int = 100,
        lr: float = 0.005,
        **kwargs: Any,
    ) -> None:
        """Initialize PLAD network.

        Args:
            classifier (BaseClassifier): the classifier that will be used at the
                end of the architecture.
            vae (nn.Module): the VAE network used to learn and generate noise.
            lambda_ (float): coefficient for the noise loss.
            optimizer (Optional[ Union[torch.optim.Optimizer, Dict[str, Any]] ],
                optional):
                a PyTorch optimizer. Defaults to None.
            require_train_stat (bool, optional): linked to the need of threshold
                calculation on the train dataset during the validation. Used by
                the schedulers to know the name of the loss to monitor. Defaults
                to False.
            epochs (int, optional): maximum number of epochs that the model will
                be trained. Used by the schedulers. Defaults to 100.
            lr (float, optional): initial learning rate. Defaults to 0.005.
        """
        super().__init__(
            params=kwargs.pop("params", locals()),
        )

        self.classifier = classifier
        self.vae = vae
        self.lambda_ = lambda_
        self.optimizer = optimizer  # TODO make it contain scheduler
        self.require_train_stat = require_train_stat
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.lr = lr

        self.optimizer_: torch.optim.Optimizer
        self.scheduler_: torch.optim.lr_scheduler._LRScheduler

    @classmethod
    def input_shape(cls) -> Tuple[int, int, int]:
        return (3, 32, 32)

    def configure_optimizers_(
        self,
        parameters: Iterable[Any],
    ) -> Union[torch.optim.Optimizer, Dict[str, Any]]:
        """Initialize the optimizers and schedulers.

        Args:
            parameters (Iterable[Any]): the parameters to update during
                training.

        Returns:
            Union[torch.optim.Optimizer, Dict[str, Any]]: the optimizers
                configuration.
        """
        if self.optimizer:
            return self.optimizer

        self.optimizer_ = torch.optim.AdamW(
            parameters,
            lr=self.lr
        )

        self.scheduler_ = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer_,
            max_lr=self.lr,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
        )
        
        return {
            "optimizer": self.optimizer_,
            "lr_scheduler": {
                "scheduler": self.scheduler_,
                "monitor": VAL_LOSS
                if not self.require_train_stat
                else f"{VAL_LOSS}/dataloader_idx_1",
                "frequency": 1,
                "interval": "step",
            },
        }

    def configure_optimizers(
        self,
    ) -> Union[torch.optim.Optimizer, Dict[str, Any]]:
        return self.configure_optimizers_(
            itertools.chain(self.classifier.parameters(), self.vae.parameters())
        )

    def log_train_state(
        self,
        loss: torch.Tensor,
        classif_loss_: torch.Tensor,
        kl_divergence_: torch.Tensor,
        noise_loss_: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        self.log(TRAIN_LOSS, loss, sync_dist=True)
        self.log(CLASSIF_LOSS, classif_loss_, sync_dist=True)
        self.log(KL_DIVERGENCE, kl_divergence_, sync_dist=True)
        self.log(NOISE_LOSS, noise_loss_, sync_dist=True)

        self.log(LEARNING_RATE, self.scheduler_.get_last_lr()[0])

        for key, value in kwargs.items():
            self.log(key, value, sync_dist=True)


class PLAD(PLADBase):
    def __init__(self, classifier: PLADClassifier, **kwargs: Any) -> None:
        super().__init__(
            classifier=classifier,
            params=kwargs.pop("params", locals()),
            **kwargs,
        )

    def _loss(
        self,
        noise_loss_: torch.Tensor,
        classif_loss_: torch.Tensor,
        kl_divergence_: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Sums the 3 losses."""
        return classif_loss_ + self.lambda_ * noise_loss_ + kl_divergence_

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
        ) = self(inputs)

        noise_loss_ = noise_loss(add_noise, mult_noise)
        classif_loss_ = classif_loss(
            classif_normal, classif_anomalies, torch.squeeze(targets)
        )
        kl_divergence_ = kl_divergence(z_mean, z_logvar)

        loss = self._loss(noise_loss_, classif_loss_, kl_divergence_)

        self.log_train_state(
            loss=loss,
            noise_loss_=noise_loss_,
            classif_loss_=classif_loss_,
            kl_divergence_=kl_divergence_,
        )

        return loss

    def validation_step(  # pylint: disable=arguments-differ
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        inputs, targets = batch
        classif_results = torch.squeeze(self.classifier(inputs), dim=1)

        loss = F.binary_cross_entropy_with_logits(classif_results, targets)

        self.log(VAL_LOSS, loss, sync_dist=True)

        return {"loss": loss, "logits": classif_results}

    def forward(  # pylint: disable=arguments-differ
        self, inputs: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass.

        Args:
            inputs (torch.Tensor): (N, VAE_in) reshapable tensor.

        Returns:
            Tuple[ torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                torch.Tensor, torch.Tensor, ]:
                additive noise, multiplicative noise, embedded mean, embedded
                variance, classification results of the normal samples,
                classification results of the pseudo-anomalies, respectively.
        """
        batch_shape = inputs.size()
        
        batch_1d = inputs.view(batch_shape[0], -1)
        entity_size = batch_1d.size()[1]

        noise, z_mean, z_logvar = self.vae(batch_1d)
        add_noise, mult_noise = torch.split(noise, entity_size, dim=1)

        anomalies = (
            inputs * mult_noise.view(batch_shape) + add_noise.view(batch_shape)
        )

        augmented_batch = torch.cat((inputs, anomalies))

        classif = torch.squeeze(self.classifier(augmented_batch), dim=1)

        classif_normal = classif[:batch_shape[0]]
        classif_anomalies = classif[batch_shape[0]:]

        return (
            add_noise,
            mult_noise,
            z_mean,
            z_logvar,
            classif_normal,
            classif_anomalies,
        )
