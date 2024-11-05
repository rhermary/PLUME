"""Train network for later use as a backbone."""

from typing import List, Optional
import argparse

from pytorch_lightning.callbacks import ModelCheckpoint, Callback, EarlyStopping
from mlflow import ActiveRun

from tasks.utils import (
    end_logs,
    init_trainer,
    start_mlflow_run,
)
from datasets.base import BaseDataModule, BaseDataset
from models.base import BaseFeatureExtractor
from metrics import MeasureAccuracy, VAL_ACC
from misc.parser import FeatureTrainingParser


def train(args: argparse.Namespace, run: Optional[ActiveRun]) -> None:
    """
    Train a feature extractor network.

    Args:
        args (argparse.Namespace): parsed arguments passed to the script
        run (ActiveRun): object with current MLFlow run info
    """
    model_checkpoint = ModelCheckpoint(monitor=VAL_ACC, mode="max")

    feature_extractor_class = BaseFeatureExtractor.get_class(args.model_name)
    datamodule = BaseDataModule[BaseDataset].select(
        name=args.dataset,
        resize=feature_extractor_class.input_shape(),
        **vars(args),
    )

    callbacks: List[Callback] = [
        MeasureAccuracy(num_classes=datamodule.num_classes),
        EarlyStopping(
            monitor=BaseFeatureExtractor.VAL_LOSS,
            min_delta=0.00,
            patience=20,
            mode="min",
            log_rank_zero_only=True,
        ),
        model_checkpoint,
    ]

    trainer, logger = init_trainer(args, callbacks=callbacks)

    model = BaseFeatureExtractor.load_or_create(
        args.model_name,
        args.model_checkpoint,
        num_classes=datamodule.num_classes,
        load_weights=args.load_weights,
    )

    trainer.fit(model, datamodule)

    end_logs(model, datamodule, args, logger, model_checkpoint, trainer=trainer, log_sample=False)


if __name__ == "__main__":
    parser = FeatureTrainingParser()
    args_ = parser.parse_args_()

    with start_mlflow_run(args_) as run_:
        train(args_, run_)
