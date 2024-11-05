"""Callback measuring per-class precision."""

from typing import Any, Dict, Literal

import mlflow
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torchmetrics import Accuracy, Metric

from metrics.threshold import DynamicThreshold
from misc.trainer import Trainer
from models.base import BaseModule


VAL_DETAILED_ACC = "val_acc_class_{}".format
THRESHOLD = "threshold"


class MeasureDetailedOneClassPrecision(Callback):
    """Measures the precision of the network for each class in the dataset.

    This callback is using the Accuracy metric, but by filtering everyclass but
    one for each metric, there is only positives (for the normal class) or
    negative samples (of the anomaly class), so the calculated metric is
    actually the precision.

    """

    def __init__(
        self,
        normal_class: int = 0,
        num_classes: int = 10,
        threshold: float = 0.5,
        dynamic_threshold: bool = False,
    ) -> None:
        super().__init__()

        self.task: Literal["binary", "multiclass", "multilabel"] = "binary"
        self.normal_class = normal_class
        self.num_classes = num_classes
        self.threshold = threshold

        self.dynamic_threshold = dynamic_threshold
        self.threshold_compute: DynamicThreshold

        self.val_accs: Dict[int, Metric]
        self.computed_val_accs: Dict[int, float] = {}
        self.original_targets: torch.Tensor

    def on_fit_start(  # type: ignore
        self, trainer: Trainer, pl_module: pl.LightningModule
    ) -> None:
        device: torch.device = pl_module.device  # type: ignore

        self.val_accs = {
            class_label: Accuracy.__new__(Accuracy, task=self.task).to(device)
            for class_label in range(self.num_classes)
        }
        self.threshold_compute = DynamicThreshold().to(device)
        self.original_targets = trainer.datamodule.val_set.targets.to(
            device, copy=True
        )

    def on_validation_batch_end(  # type: ignore[override] # pylint: disable=too-many-arguments
        self,
        trainer: Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict[str, Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if (
            outputs is None
            or not isinstance(outputs, dict)
            or self.original_targets is None
        ):
            return

        if self.dynamic_threshold:
            if dataloader_idx == 0:
                self.threshold_compute.update(outputs["logits"])
                return

            # Return cached value if already computed
            self.threshold = self.threshold_compute.compute()

        start = batch_idx * trainer.datamodule.batch_size
        original_batch_targets = self.original_targets[
            start : start + batch[1].shape[0]
        ]
        classes = original_batch_targets.unique().tolist()

        for class_label in classes:
            # Validation dataset is neither shuffled nor subset on one class
            # classification
            idx = original_batch_targets == class_label

            masked_outputs = (
                torch.sigmoid(outputs["logits"][idx]) > self.threshold
            )
            masked_targets = batch[1][idx]

            self.val_accs[class_label].update(masked_outputs, masked_targets)

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
        for class_label in range(self.num_classes):
            val_acc = self.val_accs[class_label].compute().item()

            pl_module.log(
                VAL_DETAILED_ACC(class_label),
                val_acc,
                rank_zero_only=True,
                sync_dist=True,
            )
            self.computed_val_accs[class_label] = val_acc
            self.val_accs[class_label].reset()

        pl_module.log(
            THRESHOLD, self.threshold, rank_zero_only=True, sync_dist=True
        )

        self.threshold_compute.reset()

    def state_dict(self) -> Dict[str, Any]:
        return {
            str(class_label): accuracy
            for class_label, accuracy in self.computed_val_accs.items()
        }

    @classmethod
    @rank_zero_only
    def mlflow_log(cls, state_dict: Dict[str, Any], prefix: str = "") -> None:
        """Log the calculated metrics to mlflow."""
        for class_label, accuracy in state_dict.items():
            mlflow.log_metric(
                f"{prefix}{VAL_DETAILED_ACC(class_label)}", accuracy
            )
