"""PytorchLightning trainer with more precise typing."""

import pytorch_lightning as pl

from datasets.base import BaseDataModule


class Trainer(pl.Trainer):
    """PytorchLightning trainer with compliant typing to this project."""

    datamodule: BaseDataModule
