"""Callback calculating the accuracy of the network."""

from typing import Any, Dict, Literal, Optional

import mlflow
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics import Accuracy, Metric

from models.base import BaseModule

VAL_ACC = "val_acc"


class MeasureAccuracy(Callback):
    """Callback to measure the accuracy on validation."""

    def __init__(
        self,
        task: Literal["binary", "multiclass", "multilabel"] = "multiclass",
        num_classes: int = 10,
        threshold: float = 0.5,
        require_train_stat: bool = False,
    ) -> None:
        super().__init__()

        self.task = task
        self.num_classes = num_classes
        self.threshold = threshold
        self.require_train_stat = require_train_stat

        self.val_acc: Metric
        self.computed_val_acc: float

    def on_fit_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self.val_acc = Accuracy.__new__(
            Accuracy, task=self.task, num_classes=self.num_classes
        )
        device: torch.device = pl_module.device  # type: ignore
        self.val_acc = self.val_acc.to(device)

    def on_validation_batch_end(  # pylint: disable=too-many-arguments
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if (
            (self.require_train_stat and dataloader_idx == 0)
            or outputs is None
            or not isinstance(outputs, dict)
        ):
            # TODO Threshold could be dynamically calculated also here
            return

        if self.task == "binary":
            targets = batch[1]
            # Easy dynamic threshold setting (values in [0,1] implies `sigmoid`
            # is not reapplied in the metric `update`).
            predictions = torch.sigmoid(outputs["logits"]) > self.threshold
        else:
            targets = torch.argmax(batch[1], dim=1)
            predictions = outputs["logits"]

        self.val_acc.update(predictions, targets)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: BaseModule
    ) -> None:
        """
        Called at the end of an epoch, before
        `LightningModule.on_validation_end`

        Args:
            trainer (pl.Trainer): the trainer used during training
            pl_module (pl.LightningModule): the trained model
        """
        self.computed_val_acc = self.val_acc.compute().item()

        # Synchronization is already done in `Metric.compute()`;
        # `sync_dist=True` will do the average of `n` times the same value,
        # but it removes warnings.
        pl_module.log(
            VAL_ACC, self.computed_val_acc, rank_zero_only=True, sync_dist=True
        )

        self.val_acc.reset()

    def state_dict(self) -> Dict[str, Any]:
        return {VAL_ACC: self.computed_val_acc}

    @classmethod
    @rank_zero_only
    def mlflow_log(cls, state_dict: Dict[str, Any], prefix: str = "") -> None:
        """Log the calculated metric to mlflow."""
        mlflow.log_metric(f"{prefix}{VAL_ACC}", state_dict[VAL_ACC])
