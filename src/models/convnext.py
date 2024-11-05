
from typing import Any, Union, Dict, Tuple
from functools import partial

from torchvision.models.convnext import convnext_base, ConvNeXt_Base_Weights, LayerNorm2d
import torch
from torch import nn

from .base import BaseFeatureExtractor


class ConvNeXt(BaseFeatureExtractor):
    def __init__(
        self,
        load_weights: bool = True,
        num_classes: int = 1,
        epochs: int = 100,
        steps_per_epoch: int = 100,
        lr: float = 0.005,
        **kwargs,
    ) -> None:
        super().__init__(
            params=kwargs.pop("params", locals()),
            **kwargs,
        )

        self.loss_function = nn.CrossEntropyLoss()

        if load_weights:
            self.convnext = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        else:
            self.convnext = convnext_base()

        self.lr = lr
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch

        del self.convnext.classifier
        lastconv_output_channels = 1024
        norm_layer = partial(LayerNorm2d, eps=1e-6)
        self.convnext.classifier = nn.Sequential(
            norm_layer(lastconv_output_channels),
            nn.Flatten(1),
            nn.Linear(lastconv_output_channels, num_classes)
        )

        self.optimizer_: torch.optim.Optimizer
        self.scheduler_: torch.optim.lr_scheduler._LRScheduler


    def configure_optimizers(self) -> Union[torch.optim.Optimizer, Dict[str, Any]]:
        self.optimizer_ = torch.optim.AdamW(
            self.convnext.parameters(),
            lr=self.lr,
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
                "monitor": self.VAL_LOSS,
                "frequency": 1,
                "interval": "step",
            },
        }
    
    @classmethod
    def input_shape(cls) -> Tuple[int, int, int]:
        return (3, 224, 224)

    def forward(self, inputs: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        return self.convnext(inputs)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], *args: Any, **kwargs: Any) -> torch.Tensor:
        inputs, targets = batch

        preds = self(inputs)

        loss = self.loss_function(preds, targets)

        self.log(self.TRAIN_LOSS, loss, sync_dist=True)

        return {"loss": loss}


    def validation_step(  # pylint: disable=arguments-differ
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        inputs, targets = batch
        preds = self(inputs)

        loss = self.loss_function(preds, targets)
        self.log(self.VAL_LOSS, loss, sync_dist=True)

        return {"loss": loss, "logits": preds}
