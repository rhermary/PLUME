from typing import Any, Dict, Tuple, Union
import torch
from torch import nn
from torchvision.models.resnet import resnet50, ResNet50_Weights

from .base import BaseFeatureExtractor

class ResNet50(BaseFeatureExtractor):
    def __init__(
        self,
        load_weights: bool = False,
        num_classes: int = 1000,
        **kwargs
    ) -> None:
        super().__init__(
            params=kwargs.pop("params", locals()),
            **kwargs,
        )

        if load_weights:
            self.resnet50 = resnet50(ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.resnet50 = resnet50()

        self.loss_function = nn.CrossEntropyLoss()

        in_features = self.resnet50.fc.in_features
        del self.resnet50.fc
        self.resnet50.fc = nn.Linear(in_features, num_classes)
    
    def configure_optimizers(self) -> Union[torch.optim.Optimizer, Dict[str, Any]]:
        optimizer = torch.optim.SGD(
            self.resnet50.parameters(),
            lr=0.005,  # TODO use args
            momentum=0.0,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, verbose=True, threshold=1e-4
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.VAL_LOSS,
                "frequency": 1,
            },
        }

    @classmethod
    def input_shape(cls) -> Tuple[int, int, int]:
        # https://discuss.pytorch.org/t/imagenet-pretrained-models-image-dimensions/94649
        return (3, 224, 224)

    def forward(self, inputs: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        return self.resnet50(inputs)
    
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
