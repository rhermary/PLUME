"""Base abstract Module."""

from typing import Dict, Any, Union, IO, Optional, Tuple
from abc import ABCMeta, abstractmethod
import inspect

import pytorch_lightning as pl
from lightning_fabric.utilities.types import _PATH, _MAP_LOCATION_TYPE
from torch import nn
import torch

from misc.base import Base


class BaseModule(pl.LightningModule, Base, metaclass=ABCMeta):
    """
    Abstract module with efficient hyper-parameters saving; to be used by all
    implemented modules.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        """Initialize the Module.

        Args:
            params (Dict[str, Any]): dictionary containing the arguments given
                to the child-class constructor.
        """
        super().__init__()

        self.set_hparams(params)

    def set_hparams(self, params: Dict[str, Any]) -> None:
        """Save hyper-parameters for model checkpointing.

        If a `nn.Module` object was given as an argument, do not save it as a
        hyper-parameters (very slow and heavy); rather, it will save its class
        and will be instantiated on checkpoint loading.
        TODO: make instantiation accept arguments, recursively.

        `params` is expected to contain all the arguments passed to the first
        `__init__()` call (the one of the actual class being instantiated, not
        its parents' method).
        The easiest way is to have the `**kwargs` argument at all parents'
        `__init__()` methods and add:

        ```
        super().__init__(
            ...,
            params=kwargs.pop("params", locals()),
            **kwargs,
        )
        ```

        to the `super()` calls of classes subclassing `BaseModule`.

        If the original class constructor also has a `**kwargs` argument, will
        take care of it before saving the arguments, as otherwise with the given
        methodology it will be to much embedded.

        Args:
            params (Dict[str, Any]): arguments given to the child class.
        """
        signature = list(
            map(
                lambda x: x.name,
                inspect.signature(self.__class__).parameters.values(),
            )
        )
        hparams = {
            param: (
                params[param].__class__
                if isinstance(params[param], nn.Module)
                else params[param]
            )
            for param in signature
            if param != "kwargs"
        }

        if "kwargs" in params and "kwargs" in signature:
            hparams |= params["kwargs"]

        self._set_hparams(hparams)

    def get_modules(self) -> Dict[str, Optional[nn.Module]]:
        """Return the submodules of the current module.

        Returns:
            Dict[str, Optional[nn.Module]]: the module's submodules.
        """
        return self._modules

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[_PATH, IO],
        map_location: _MAP_LOCATION_TYPE = None,
        hparams_file: Optional[_PATH] = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> "BaseModule":
        """
        Overrides the classic loading function to instantiate parameters when
        needed (e.g. if they are modules, not to save the weights as
        hyperparameters).

        Checks if the saved hyper parameters are classes, if it's the case,
        instantiates them and passes them in the `**kwargs` to the original
        `load_from_checkpoint()` to override the saved arguments.
        """
        ckpt = torch.load(
            checkpoint_path, map_location=lambda storage, _: storage
        )
        params = {
            x.name: ckpt["hyper_parameters"][x.name]()
            for x in inspect.signature(cls).parameters.values()
            if x.name != "kwargs"
            and x.name in ckpt["hyper_parameters"]
            and isinstance(ckpt["hyper_parameters"][x.name], type)
        }

        del ckpt

        return super().load_from_checkpoint(
            checkpoint_path,
            map_location,
            hparams_file,
            strict,
            **(params | kwargs),
        )

    @classmethod
    def load_or_create(
        cls,
        model_name: str,
        checkpoint_path: Optional[str],
        *args: Any,
        **kwargs: Any,
    ) -> "BaseModule":
        """Load a model checkpoint or initialize a new one.

        Args:
            model_name (str): the class name of the model to load.
            checkpoint_path (Optional[str]): a path to a model checkpoint.

        Raises:
            ValueError: if the given model name does not correspond to any
                known `BaseModule` subclass.

        Returns:
            BaseModule: the loaded or newly created model.
        """
        model_class = cls.select_(model_name)
        if model_class is None:
            raise ValueError("The selected model was not found.")

        if checkpoint_path:
            return model_class.load_from_checkpoint(
                checkpoint_path, *args, **kwargs
            )

        return model_class(*args, **kwargs)

    @abstractmethod
    def configure_optimizers(
        self,
    ) -> Union[torch.optim.Optimizer, Dict[str, Any]]:
        pass

    @abstractmethod
    def forward(  # pylint: disable=arguments-differ
        self, inputs: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Any:
        pass

    @abstractmethod
    def training_step(  # pylint: disable=arguments-differ
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        pass

    @classmethod
    @abstractmethod
    def input_shape(cls) -> Tuple[int, int, int]:
        """Return the shape of the input accepted by the network.

        Returns:
            Tuple[int, int, int]: the 3D dimensions of the accepted images in
                (C, H, W) order.
        """


class BaseFeatureExtractor(BaseModule, metaclass=ABCMeta):
    """
    Abstract class used to group the features extractors under one common
    structure, mainly for the `.select()` method to work properly.
    """

    VAL_LOSS = "val_loss"
    TRAIN_LOSS = "train_loss"


class BaseAdaptativeFeatureExtractor(BaseFeatureExtractor):
    adaptor : nn.Module
