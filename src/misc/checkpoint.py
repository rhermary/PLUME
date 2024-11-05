import torch
import pytorch_lightning as pl
from pytorch_lightning.trainer import call
from pytorch_lightning.callbacks import ModelCheckpoint

class LightModelCheckpoint(ModelCheckpoint):
    def __init__(self, **kwargs):
        self.save_weights_only_ = True

        super().__init__(save_weights_only=self.save_weights_only_, **kwargs)

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        super()._save_checkpoint(trainer=trainer, filepath=filepath)

        if self.save_weights_only_:
            ckpt = torch.load(filepath)
            ckpt["callbacks"] = call._call_callbacks_state_dict(trainer)
            torch.save(ckpt, filepath)
