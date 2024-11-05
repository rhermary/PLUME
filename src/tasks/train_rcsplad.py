"""Classic PLAD training."""

import argparse
from typing import Optional
import math

import mlflow
from mlflow import ActiveRun
from pytorch_lightning.tuner import Tuner

from misc.parser import CSPLADTrainingParser
from misc.checkpoint import LightModelCheckpoint
from metrics import (
    MeasureAUC,
    MeasureAccuracy,
    MeasureDetailedOneClassPrecision,
    VAL_AUC,
)
from models import VAE, RotCosSimPLAD, BaseClassifier
from datasets.base import BaseOneClassDataModule, BaseDataset
from tasks.utils import init_trainer, start_mlflow_run, end_logs, init_trainer_


def train(args: argparse.Namespace, run: Optional[ActiveRun]) -> None:
    """Train PLAD like in the paper.

    Args:
        args (argparse.Namespace): parsed arguments passed to the script
        run (Optional[ActiveRun]): object with current MLFlow run info
    """
    model_checkpoint = LightModelCheckpoint(monitor=VAL_AUC, mode="max")

    datamodule = BaseOneClassDataModule[BaseDataset].select(
        name=args.dataset,
        resize=RotCosSimPLAD.input_shape(),
        require_train_stat=True,
        **vars(args),
    )
    datamodule.prepare_data()
    datamodule.setup()

    callbacks = [
        MeasureAUC(require_train_stat=True),
        MeasureAccuracy(task="binary", require_train_stat=True),
        MeasureDetailedOneClassPrecision(
            normal_class=args.normal_class,
            num_classes=datamodule.dataset_class.num_classes(),
            dynamic_threshold=True,
        ),
        model_checkpoint,
    ]

    classifier = BaseClassifier.select(args.classifier)
    vae = VAE()
    model = RotCosSimPLAD(
        classifier=classifier,
        vae=vae,
        lambda_=args.lambda_,
        epochs=args.epochs,
        steps_per_epoch=math.ceil(datamodule.steps_per_epoch / args.acc_grad),
        lr=args.lr if args.lr > 0.0 else 0.0,
        eta=args.eta,
        nu=args.nu,
        gamma=args.gamma,
        phi=args.phi,
        tau=args.tau,
    )

    if args.lr < 0.0:
        blank_trainer = init_trainer_(args)
        tuner = Tuner(blank_trainer)
        lr_finder = tuner.lr_find(
            model, num_training=100, datamodule=datamodule, update_attr=True
        )
        mlflow.log_param("found_lr", lr_finder.suggestion())

    trainer, logger = init_trainer(args, callbacks=callbacks)
    trainer.fit(model, datamodule)

    end_logs(
        model,
        datamodule,
        args,
        logger,
        model_checkpoint,
        check_trace=False,
        log_sample=False,
    )


if __name__ == "__main__":
    parser = CSPLADTrainingParser()
    args_ = parser.parse_args_()

    with start_mlflow_run(args_) as run_:
        train(args_, run_)
