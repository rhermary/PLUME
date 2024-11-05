"""Parsers for the different tasks."""

import os
import argparse
from typing import Any
from pathlib import Path

from models import FEATURE_EXTRACTORS, CLASSIFIERS
from models.backbones import BACKBONES
from datasets import DATASETS, ONE_CLASS_DATASETS, PARTIAL_DATASETS, MULTINORMAL_DATASETS


class Parser(argparse.ArgumentParser):
    """Parser for common arguments between tasks."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize default arguments."""
        super().__init__(**kwargs)

        self.add_argument(
            "--base_dir", default=".", help="root directory for saving outputs"
        )
        self.add_argument(
            "--dataset_dir",
            default="data",
            help="root directory where datasets are",
        )
        self.add_argument(
            "--model_checkpoint",
            default=None,
            help="path to a checkpoint",
            type=str,
        )
        self.add_argument(
            "--devices",
            default=-1,
            help="which GPU(s) to train on. Default: all GPUs",
            type=int,
            nargs="+",
        )
        self.add_argument(
            "--experiment", default="Default", help="MLFlow experiment name"
        )
        self.add_argument("--debug", action="store_true")
        self.add_argument(
            "--log_every_n_steps",
            default=50,
            help="logging interval",
            type=int,
        )
        self._dataset_action = self.add_argument(
            "--dataset",
            default="fmnist",
            choices=DATASETS,
            help="name of the dataset to use",
        )
        self.add_argument("--log_graph", action="store_true")

    def parse_args_(self) -> argparse.Namespace:
        """
        Parse the arguments given, replacing the `base_dir` option's value
        by an absolute path.
        """
        args_ = super().parse_args()
        args_.base_dir = (
            (Path(args_.base_dir or os.getcwd())).absolute().as_posix()
        )

        return args_


class TrainingParser(Parser):
    """Parser for common training arguments."""

    def __init__(self, **kwargs: Any) -> None:
        """Base arguments definition."""
        super().__init__(**kwargs)

        self.add_argument(
            "--batch_size",
            type=int,
            default=512,
            metavar="N",
            help="batch size for training",
        )
        self.add_argument(
            "--epochs",
            type=int,
            default=100,
            metavar="N",
            help="number of epochs to train",
        )
        self.add_argument(
            "--lr",
            type=float,
            default=0.005,
            metavar="LR",
            help="learning rate to use. If negative, will calculate the"
                 "starting learning rate with a LR finder",
        )
        self.add_argument(
            "--optim",
            type=int,
            default=1,
            metavar="N",
            help="[Not Supported] 0: Adam, 1: SGD",
        )
        self.add_argument(
            "--mom", type=float, default=0.0, metavar="M", help="momentum"
        )
        self.add_argument("--metric", type=str, default="AUC")
        self.add_argument(
            "--acc_grad",
            type=int,
            default=1,
            metavar="N",
            help="number of gradient batches to accumulate",
        )
        self.add_argument(
            "--dataset_limit",
            type=float,
            default=1.0,
            metavar="N",
            help="amount of the dataset used for the training",
        )
        self.add_argument(
            "--strategy",
            type=str,
            default="auto",
            help="parallel computing strategy",
        )
        self.add_argument(
            "--num_workers",
            type=int,
            default=0,
            help="number of workers to use for data loading",
        )


class PLADTrainingParser(TrainingParser):
    """Parser for base PLAD training."""

    def __init__(self, **kwargs: Any) -> None:
        """Configure the parser."""
        super().__init__(**({"description": "PLAD Training"} | kwargs))

        self.normal_class = self.add_argument(
            "--normal_class",
            type=int,
            default=5,
            metavar="N",
            help="CIFAR10 normal class index",
        )
        self.add_argument(
            "--lambda",
            type=float,
            default=5,
            metavar="N",
            help="Weight of the perturbator loss",
            dest="lambda_",
        )

        self.add_argument(
            "--classifier",
            type=str,
            default="pladclassifier",
            help="Classifier model to use",
            choices=CLASSIFIERS,
        )

        self.add_argument(
            "--force_grayscale",
            action="store_true",
            help="If the dataset is RGB, forces the use of only the luminance"
            "channel regardless of the number of required input channels"
            "by the network. If necessary, will duplicate the channel.",
        )

        self._dataset_action.default = "oneclassfmnist"
        self._dataset_action.choices = ONE_CLASS_DATASETS


class CSPLADTrainingParser(PLADTrainingParser):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_argument(
            "--eta",
            type=float,
            default=0.0,
            metavar="N",
            help="Weight of the cosine similarity constraint loss",
        )
        self.add_argument(
            "--nu",
            type=float,
            default=0.0,
            metavar="N",
            help="Weight of the cosine similarity coercion loss",
        )
        self.add_argument(
            "--gamma",
            type=float,
            default=0.0,
            metavar="N",
            help="Weight of the contrastive loss",
        )
        self.add_argument(
            "--phi",
            type=float,
            default=0.0,
            metavar="N",
            help="Weight of the mean contrastive loss",
        )
        self.add_argument(
            "--tau",
            type=float,
            default=1.0,
            metavar="N",
            help="Value of the temperature parameter",
        )


class FeatureTrainingParser(TrainingParser):
    """Parser for the feature extractor training."""

    def __init__(self) -> None:
        """Configure the parser."""
        super().__init__(description="Feature Extractor Training")

        self.add_argument("--load_weights", action="store_true")
        self.add_argument(
            "--model_name",
            choices=FEATURE_EXTRACTORS,
            default="resnet50",
            help="the model architecture to train",
        )

    def parse_args_(self) -> argparse.Namespace:
        args_ = super().parse_args_()

        if args_.load_weights and not args_.model_checkpoint is None:
            raise ValueError(
                "'--load_weights' and '--model_checkpoint' are "
                "mutually exclusive options"
            )

        return args_


class PartialFeatureTrainingParser(FeatureTrainingParser):
    """Parser for feature extractor training with a partial dataset."""

    def __init__(self) -> None:
        """Configure the parser."""
        super().__init__()

        self.add_argument(
            "--excluded_classes",
            default=[0],
            help="which classes to exclude from the training",
            type=int,
            nargs="+",
        )

        self._dataset_action.default = "partialfmnist"
        self._dataset_action.choices = PARTIAL_DATASETS


class CombinedTrainingParser(PLADTrainingParser):
    """Parser for the combined network training."""

    def __init__(self) -> None:
        """Configure the parser."""
        super().__init__(description="Feature Extractor Training")

        self.add_argument(
            "--feature_extractor_checkpoint",
            default=None,
            help="path to pretrained feature extractor",
            type=str,
        )

        self.add_argument(
            "--backbone_name",
            help="name of the backbone associated to the pretrained model",
            choices=BACKBONES,
            type=str,
        )

        self.add_argument(
            "--features_dir",
            help="directory of the saved extracted features (dataset_dir subpath)",
            type=str,
        )

    def parse_args_(self) -> argparse.Namespace:
        args_ = super().parse_args_()

        if (
            args_.feature_extractor_checkpoint is None
            and args_.features_dir is not None
        ):
            raise ValueError(
                "Path to the checkpoint used to generate the features is "
                "mandatory. Provide it with the "
                "'--feature_extractor_checkpoint' option"
            )

        return args_


class CSCombinedTrainingParser(CombinedTrainingParser):
    def __init__(self) -> None:
        """Configure the parser."""
        super().__init__()

        self.add_argument(
            "--eta",
            type=float,
            default=0.0,
            metavar="N",
            help="Weight of the cosine similarity constraint loss",
        )
        self.add_argument(
            "--nu",
            type=float,
            default=0.0,
            metavar="N",
            help="Weight of the cosine similarity coercion loss",
        )
        self.add_argument(
            "--gamma",
            type=float,
            default=0.0,
            metavar="N",
            help="Weight of the contrastive loss",
        )
        self.add_argument(
            "--phi",
            type=float,
            default=0.0,
            metavar="N",
            help="Weight of the mean contrastive loss",
        )
        self.add_argument(
            "--tau",
            type=float,
            default=1.0,
            metavar="N",
            help="Value of the temperature parameter",
        )


class PLADMultiTrainingParser(PLADTrainingParser):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self._remove_action(self.normal_class)
        
        self.add_argument(
            "--normal_classes",
            default=[0],
            type=int,
            nargs="+",
        )

        self._dataset_action.default = "multinormalcifar10"
        self._dataset_action.choices = MULTINORMAL_DATASETS
