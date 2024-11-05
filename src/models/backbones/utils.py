from models.backbones.base import BaseBackbone
from models.base import BaseFeatureExtractor

from lightning_fabric.utilities.types import _PATH

from typing import IO, Any, Type, Union


def load_from_checkpoint_(
    backbone_class: Type[BaseBackbone],
    network_class: Type[BaseFeatureExtractor],
    checkpoint_path: Union[_PATH, IO],
    *args: Any,
    **kwargs: Any,
) -> "BaseBackbone":
    """Load the trained model and transfer its weights to the backbone model.

    `kwargs` can be used to overwrite the checkpoint parameters given to the
    model constructor.

    Args:
        backbone_class (Type[BaseBackbone]): the class of the backbone linked to
            the class of the trained model to load.
        network_class (Type[BaseFeatureExtractor]): the class of the trained
            model.
        checkpoint_path (Union[_PATH, IO]): the path of the saved trained model.

    Returns:
        BaseBackbone: the backbone initialized with the trained model
            weights.
    """
    ckpt = network_class.load_from_checkpoint(checkpoint_path, *args, **kwargs)
    model: BaseBackbone = backbone_class(**(ckpt.hparams | kwargs))
    model.transfer_modules(ckpt)

    del ckpt
    model.post_init_()

    return model