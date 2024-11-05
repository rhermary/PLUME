"""Classic PLAD training."""

import argparse
from typing import Type
import math

import mlflow
from mlflow import ActiveRun
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.tuner import Tuner

from misc.parser import CSCombinedTrainingParser
from models import (
    PLAD,
    VAE,
    FeatureCSPLAD,
    AdaptativeFeatureCSPLAD,
    BaseClassifier,
    BaseAdaptativeFeatureExtractor,
)

from models.backbones.base import BaseBackbone
from models.backbones import ResNet50Backbone, VGG16Backbone
from metrics import (
    MeasureAUC,
    VAL_AUC,
    MeasureAccuracy,
    MeasureDetailedOneClassPrecision,
)
from datasets.base import BaseOneClassDataModule, BaseDataset
from tasks.utils import (
    init_trainer,
    init_trainer_,
    start_mlflow_run,
    end_logs,
)


def choose_model(backbone: BaseBackbone) -> Type[PLAD]:
    """
    Chose PLAD adaptation depending on if the feature extractor already has
    a dense layer with the correct output dimension.
    """
    if isinstance(backbone, ResNet50Backbone) or isinstance(backbone, VGG16Backbone):
        return AdaptativeFeatureCSPLAD
    
    if getattr(backbone, "dense_adaptor", None) is None:
        # return FeaturePLAD
        raise NotImplementedError("Legacy Code")

    return FeatureCSPLAD


def train(args: argparse.Namespace, run: ActiveRun) -> None:
    """Train PLAD with pretrained backbone.

    Args:
        args (argparse.Namespace): parsed arguments passed to the script
        run (ActiveRun): object with current MLFlow run info
    """
    model_checkpoint = ModelCheckpoint(monitor=VAL_AUC, mode="max")

    features_given_directly=args.features_dir is not None

    backbone_class = BaseBackbone.get_class(args.backbone_name)
    datamodule = BaseOneClassDataModule[BaseDataset].select(
        name=args.dataset,
        resize=backbone_class.input_shape()
            if not features_given_directly
            else None,
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

    backbone = BaseBackbone.load(
        args.backbone_name,
        args.feature_extractor_checkpoint,
    )
    classifier = BaseClassifier.select(args.classifier)
    vae = VAE()
    model = choose_model(backbone)(
        classifier=classifier,
        vae=vae,
        lambda_=args.lambda_,
        epochs=args.epochs,
        steps_per_epoch=math.ceil(datamodule.steps_per_epoch / args.acc_grad),
        lr=args.lr if args.lr > 0.0 else 0.0,
        backbone=backbone,
        features_given_directly=features_given_directly,
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
        check_trace=True,
        log_sample=False,
    )


if __name__ == "__main__":
    parser = CSCombinedTrainingParser()
    args_ = parser.parse_args_()

    with start_mlflow_run(args_) as run_:
        train(args_, run_)
