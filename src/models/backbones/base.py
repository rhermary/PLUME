"""Base structure of all backbones."""

from abc import abstractmethod, ABCMeta
from typing import Any, Dict, Union, IO

from lightning_fabric.utilities.types import _PATH

from models.base import BaseModule


class MustSubclassAnotherBaseModule(ABCMeta):
    """Metaclass checking that the `BaseBackbone` class is correctly used.

    A backbone in the context of this project is a network used for feature
    extraction (features then used in an anomaly detection module).

    The backbone is a modified version of a trained model and thus needs to
    extend (subclass) another model class (which subclasses `BaseModule`).
    """

    def __new__(
        mcs,
        __name: str,
        __bases: tuple[type, ...],
        __namespace: dict[str, Any],
        **kwargs: Any,
    ) -> "MustSubclassAnotherBaseModule":
        bases = [list(__bases)]
        while __name != "BaseBackbone" and len(bases) > 0:
            bases_ = bases.pop()

            bases_names = [base.__name__ for base in bases_]
            if "BaseBackbone" in bases_names[0]:
                if not len(bases_) > 1:
                    raise AssertionError(
                        "`BaseBackbone` class must always be inherited with "
                        "another trainable model class, inheriting `BaseModule`."
                    )

                break

            for base in bases_:
                bases.append(list(base.__bases__))

        return super().__new__(mcs, __name, __bases, __namespace, **kwargs)


class BaseBackbone(BaseModule, metaclass=MustSubclassAnotherBaseModule):
    """Interface for backbones, defining common functions.

    Subclassing `BaseModule` is mandatory as the `transfer_modules` function
    needs the `torch.nn.Module._modules` attribute. Static typing requires the
    type definition of the latter; redefining its type is way easier, but does
    not seem to be a good practice solution.
    """

    def __init__(  # pylint: disable=super-init-not-called
        self, params: Dict[str, Any]
    ) -> None:
        """Bypass `BaseModule.__init__`.

        This method should not be called, as the `BaseModule` subclass
        constructor should be called directly.
        """

    @abstractmethod
    def post_init_(self) -> None:
        """Function called at the end of `__init__()` to remove unwanted layers.

        This is useful to be able to remove the correct layers in the
        checkpoint loading phase.
        """

    def transfer_modules(self, other_module: "BaseModule") -> None:
        """Replace the current weights with the one from the given module.

        Args:
            other_module (BaseModule): loaded network from which to take the
                weights.
        """
        del self._modules
        self._modules = other_module.get_modules()

    @classmethod
    def load(
        cls, backbone_name: str, checkpoint_path: Union[_PATH, IO]
    ) -> "BaseBackbone":
        """Load the weights of the trained network and initialize the backbone.

        Args:
            backbone_name (str): the name of the backbone class to load. It is
                related to the class of the trained model to load.
            checkpoint_path (Union[_PATH, IO]): the path of the saved model.

        Raises:
            ValueError: if the given `backbone_name` does not correspond to any
                known class.

        Returns:
            BaseBackbone: the backbone initialized with the trained model
                weights.
        """
        backbone_class = cls.select_(backbone_name)
        if backbone_class is None:
            # Should never happen as the name is checked by the argument parser
            raise ValueError("Selected backbone class is not available")

        return backbone_class.load_from_checkpoint(checkpoint_path, load_weights=False)
