"""Defines the base class for all models and datasets."""

from abc import ABCMeta, abstractmethod
import inspect
from typing import Optional, Type, Any, List, TypeVar

from pytorch_lightning.utilities.rank_zero import rank_zero_only
import mlflow

class Base(metaclass=ABCMeta):
    """Abstract class mainly defining the `select` and `listing` methods."""

    @abstractmethod
    def __init__(self) -> None:
        pass

    @classmethod
    def select_(cls: Type["B"], name: str) -> Optional[Type["B"]]:
        """If exists in the subclasses, returns the desired class."""
        if cls.__name__.lower() == name:
            return cls

        for subclass in cls.__subclasses__():
            selected = subclass.select_(name)
            if selected is not None:
                return selected

        return None

    @classmethod
    def get_class(cls: Type["B"], name: str) -> Type["B"]:
        """If exists in the subclasses, returns the desired class.

        Raises:
            ValueError: if the name does not correspond to any subclass.

        """
        selected = cls.select_(name)
        if selected is None:
            raise ValueError("The selected class was not found.")

        return selected

    @classmethod
    def select(cls: Type["B"], name: str, **kwargs: Any) -> "B":
        """If exists in the subclasses, returns the instantiated object.

        Raises:
            ValueError: if the name does not correspond to any subclass.

        """
        selected = cls.get_class(name)

        return selected(**kwargs)

    @classmethod
    def listing(cls: Type["B"]) -> List[str]:
        """Lists the detected subclasses."""
        subclasses = set()
        if not inspect.isabstract(cls):
            subclasses = {cls.__name__.lower()}

        for subclass in cls.__subclasses__():
            subclasses = subclasses.union(subclass.listing())

        return list(subclasses)
    
    @classmethod
    @rank_zero_only
    def log_param(cls: Type["B"], key: str, value: Any) -> None:
        if mlflow.active_run() is not None:
            mlflow.log_param(key, value)



# Creating this type variable allows better static sub-typing/type resolution
# in the subclasses' methods.
B = TypeVar("B", bound=Base)
