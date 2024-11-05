"""Callback calculating the AUC on the validation dataset."""

from typing import Any, Dict, Literal, Optional

import mlflow
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics import AUROC, Metric

from models.plad import PLAD


VAL_AUC = "val_auc"


class MeasureAUC(Callback):
    """Callback to measure the AUC on validation."""

    def __init__(
        self,
        task: Literal["binary", "multiclass", "multilabel"] = "binary",
        num_classes: Optional[int] = None,
        require_train_stat: bool = False,
    ) -> None:
        super().__init__()

        self.task = task
        self.require_train_stat = require_train_stat

        self.num_classes = num_classes
        self.val_auc: Metric
        self.computed_val_auc: float

    def on_fit_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self.val_auc = AUROC.__new__(
            AUROC, task=self.task, num_classes=self.num_classes
        )
        device: torch.device = pl_module.device  # type: ignore
        self.val_auc = self.val_auc.to(device)

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
            return

        self.val_auc.update(outputs["logits"], batch[1])

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: PLAD
    ) -> None:
        """
        Called at the end of an epoch, before
        `LightningModule.on_validation_end`

        Args:
            trainer (pl.Trainer): the trainer used during training
            pl_module (pl.LightningModule): the trained model
        """

        # Automatically gather metrics from all GPUs if in a distributed session
        self.computed_val_auc = self.val_auc.compute().item()

        pl_module.log(VAL_AUC, self.computed_val_auc, rank_zero_only=True)
        self.val_auc.reset()

    def state_dict(self) -> Dict[str, Any]:
        return {VAL_AUC: self.computed_val_auc}

    @classmethod
    @rank_zero_only
    def mlflow_log(cls, state_dict: Dict[str, Any], prefix: str = "") -> None:
        """Log the calculated metric to mlflow."""
        mlflow.log_metric(f"{prefix}{VAL_AUC}", state_dict[VAL_AUC])
