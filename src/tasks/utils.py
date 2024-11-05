"""Utility functions for the `tasks` module."""

from typing import Optional, Union, Tuple, Any, Generator
import argparse
from pathlib import Path
import contextlib
import sys
import secrets
import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import mlflow
from mlflow import ActiveRun
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import torch

from datasets.base import BaseDataModule
from metrics import (
    MeasureAUC,
    MeasureAccuracy,
    MeasureDetailedOneClassPrecision,
)
from misc.context import patch_over_check_trace
from misc.trainer import Trainer
from misc.checkpoint import LightModelCheckpoint


@rank_zero_only
def _create_logger(
    args: argparse.Namespace,
    activate_log_graph: bool = False,
    version: Optional[Union[int, str]] = None,
) -> TensorBoardLogger:
    """Creates the TensorBoard logger based on the given arguments."""
    # Trying to mitigate the logs overwriting when jobs are ran in parallel
    xp_path = Path(args.base_dir) / "lightning_logs" / args.experiment
    while version is None or os.path.isdir(xp_path / version):
        version = "-".join(
            map(
                lambda x: str(x).zfill(4),
                [
                    len(os.listdir(xp_path)) if os.path.isdir(xp_path) else 0,
                    secrets.randbelow(1000),
                    args.normal_class
                    if hasattr(args, "normal_class")
                    else secrets.randbelow(1000),
                ],
            )
        )

    logger = TensorBoardLogger(
        Path(args.base_dir) / "lightning_logs",
        default_hp_metric=True,
        name=args.experiment,
        log_graph=activate_log_graph,
        version=version,
    )

    mlflow.log_param("TB_folder", logger.log_dir)
    logger.log_hyperparams(args)

    return logger


@rank_zero_only
def log_graph(
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    args: argparse.Namespace,
    version: Union[int, str],
    check_trace: bool = True,
) -> None:
    """Log the graph of the given model to TensorBoard.

    Args:
        model (pl.LightningModule): model to be traced
        datamodule (pl.LightningDataModule): datamodule containing sample data
        args (argparse.Namespace): arguments passed to the script; used to create logger
        version (Union[int, str]): version (id) of the current logger, if any. Defaults to None.
        check_trace (bool, optional): enable outputs comparison during tracing. Defaults to True.
    """
    logger = _create_logger(args, True, version)
    model.eval()
    model.cpu()

    if check_trace:
        logger.log_graph(model, next(iter(datamodule.val_dataloader()))[0][:10])
        return

    with patch_over_check_trace():
        logger.log_graph(model, next(iter(datamodule.val_dataloader()))[0][:10])

def init_trainer_(
    args: argparse.Namespace,
    **kwargs: Any,
) -> Trainer:
    trainer = Trainer(
        devices=args.devices,
        accelerator="gpu",
        strategy=args.strategy,
        fast_dev_run=args.debug,
        max_epochs=args.epochs,
        accumulate_grad_batches=args.acc_grad,
        log_every_n_steps=args.log_every_n_steps,
        reload_dataloaders_every_n_epochs=1,
        **kwargs,
    )

    return trainer


def init_trainer(
    args: argparse.Namespace,
    **kwargs: Any,
) -> Tuple[Trainer, TensorBoardLogger]:
    """Initialize the PytorchLightning trainer (and loggers).

    To avoid MLFlow logging duplicates, only initialize this logger on the GPU
    of rank 0.

    """
    tb_logger = _create_logger(args)
    loggers = tb_logger

    rank = getattr(rank_zero_only, "rank", -1)
    if rank == 0:
        mlf_logger = MLFlowLogger(
            experiment_name=args.experiment,
            tracking_uri=mlflow.get_tracking_uri(),
            run_id=mlflow.active_run().info.run_id,
        )
        loggers = [tb_logger, mlf_logger]

    trainer = init_trainer_(args, logger=loggers, **kwargs)

    return trainer, tb_logger


@contextlib.contextmanager
def start_mlflow_run(
    args_: argparse.Namespace,
) -> Generator[Optional[ActiveRun], None, None]:
    """Starts a MLFlow run, also logging the given program arguments.

    Args:
        args_ (argparse.Namespace): the arguments passed to the program.

    Yields:
        Generator[Optional[ActiveRun], None, None]: None if the process is not
            if the GPU of rank 0, else the started MLFlow run.
    """
    rank = getattr(rank_zero_only, "rank", -1)
    if rank != 0:
        yield None
        return

    tracking_uri = (Path(args_.base_dir) / "mlruns").as_posix()
    mlf_logger = MLFlowLogger(
        experiment_name=args_.experiment,
        tracking_uri=tracking_uri,
    )

    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run(mlf_logger.run_id) as run_:
        mlflow.log_params(vars(args_))
        mlflow.log_param("python_path", sys.executable)

        yield run_


@rank_zero_only
def log_best(model_checkpoint: ModelCheckpoint) -> None:
    """Log the best model path and the associated metrics to MLFlow."""
    if not model_checkpoint.best_model_path:
        return

    callbacks = torch.load(model_checkpoint.best_model_path)["callbacks"]

    for callback_name, state_dict in callbacks.items():
        match callback_name:
            case MeasureDetailedOneClassPrecision.__name__:
                MeasureDetailedOneClassPrecision.mlflow_log(
                    state_dict, prefix="best_"
                )
            case MeasureAccuracy.__name__:
                MeasureAccuracy.mlflow_log(state_dict, prefix="best_")
            case MeasureAUC.__name__:
                MeasureAUC.mlflow_log(state_dict, prefix="best_")
            case name if name.startswith(ModelCheckpoint.__name__) or name == LightModelCheckpoint.__name__:
                mlflow.log_param(
                    "best_model_path", model_checkpoint.best_model_path
                )
            case _:
                continue


@rank_zero_only
def plot_grid(samples: np.ndarray) -> np.ndarray:
    """Plot grid image with given samples. Return the figure as an image."""

    samples = samples.astype(np.float32)
    samples -= samples.min()
    samples /= samples.max()

    fig, axes = plt.subplots(2, 5, figsize=(15, 5))
    for sample, axis in zip(samples, axes.reshape(-1)):
        axis.invert_yaxis()
        axis.get_yaxis().set_ticks([])
        axis.get_xaxis().set_ticks([])
        axis.set_box_aspect(1)

        sample_ = sample.squeeze()
        if sample_.shape[0] == 3:
            sample_ = np.moveaxis(sample_, 0, -1)

        plot = axis.pcolormesh(sample_, cmap="plasma")
        divider = make_axes_locatable(axis)
        clb_ax = divider.append_axes("right", size="5%", pad=0.05)
        clb_ax.set_box_aspect(15)
        plt.colorbar(plot, cax=clb_ax)

    fig.canvas.draw()
    plt.close(fig)

    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return image.reshape(fig.canvas.get_width_height()[::-1] + (3,))


@rank_zero_only
def end_logs(  # pylint: disable=too-many-arguments
    model: pl.LightningModule,
    datamodule: BaseDataModule,
    args: argparse.Namespace,
    logger: TensorBoardLogger,
    model_checkpoint: ModelCheckpoint,
    log_sample: bool = False,
    trainer: pl.Trainer = None,
    check_trace: bool = True,
) -> None:
    """
    Log best model path and metrics in MLFlow; model's trace and samples if
    asked.
    """
    if args.log_graph:
        log_graph(
            model,
            datamodule,
            args,
            version=logger.version,
            check_trace=check_trace,
        )

    if log_sample:
        sample = datamodule.sample[0]
        if isinstance(sample, torch.Tensor):
            sample = sample.detach().cpu().numpy()

        grid = plot_grid(sample)

        logger.experiment.add_image(
            "Input Samples",
            grid,
            dataformats="HWC",
        )

        mlflow.log_image(grid, "input_samples.png")

        sample = datamodule.sample_transformed[0]
        if isinstance(sample, torch.Tensor):
            sample = sample.detach().cpu().numpy()

        grid = plot_grid(sample)

        logger.experiment.add_image(
            "Transformed Samples",
            grid,
            dataformats="HWC",
        )

        mlflow.log_image(grid, "transformed_samples.png")

    log_best(model_checkpoint)
    if not trainer is None and args.epochs == 0:
        trainer.save_checkpoint(f"{logger.log_dir}/checkpoints/start.ckpt")
