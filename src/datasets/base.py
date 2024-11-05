"""Base classes for the different datasets and objects."""

from abc import ABCMeta, abstractmethod
from typing import (
    Any,
    Tuple,
    Union,
    Optional,
    List,
    Type,
    TypeVar,
    Generic,
)
import math

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Subset as TorchSubset
from torch import Tensor
from torchvision import transforms
import numpy as np

from misc.base import Base


class BaseDataset(TorchDataset, metaclass=ABCMeta):
    """Base dataset abstract class."""

    # TODO add function mapping category index to category name
    def __init__(self, limit: float) -> None:
        super().__init__()

        self._limit = limit

    def __len__(self) -> int:
        # Forcing a limit of seeable samples, useful to do quick tasks and
        # debugging
        return int(self.full_length * self._limit)

    @property
    @abstractmethod
    def full_length(self) -> int:
        """True length of the dataset (not limited)."""

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        pass

    @property
    @abstractmethod
    def inputs(self) -> Tensor:
        """Raw data of the dataset.

        Returns:
            Tensor: All of the data if available/possible, otherwise part of it.
        """

    @property
    @abstractmethod
    def targets(self) -> Tensor:
        """Raw targets of the dataset.

        Returns:
            Tensor: All are part of the desired outputs.
        """

    @classmethod
    def prepare_data(cls, *args: Any, **kwargs: Any) -> None:
        """Function to be called in `LightningModule.prepare_data()`

        This function is useful to specify how to download the data for
        example.
        """

    @classmethod
    @abstractmethod
    def num_classes(cls) -> int:
        """Return the original number of classes in the data set.

        Returns:
            int: the number of classes.
        """
        
    @classmethod
    @abstractmethod
    def original_shape(cls) -> Tuple[int, int, int]:
        pass

    @classmethod
    @abstractmethod
    def mean(cls) -> np.ndarray:
        """Return the mean of the dataset.

        Can be the mean over the whole dataset, the mean of each class, and even
        the mean per channel (or per class and per channel).

        Returns:
            Tensor: the mean(s).
        """

    @classmethod
    @abstractmethod
    def std(cls) -> np.ndarray:
        """Return the standard deviation of the dataset.

        Can be the std over the whole dataset, the std of each class, and even
        the std per channel (or per class and per channel).

        Returns:
            Tensor: the standard deviation(s).
        """


DatasetClass = TypeVar("DatasetClass", bound=BaseDataset)


class Subset(TorchSubset, Generic[DatasetClass]):
    """Subclasses torch.utils.data.Subset for typing purposes.

    Also defines "bridges" to `BaseDataset` methods for compatibility purposes.
    """

    dataset: DatasetClass

    @property
    def full_length(self) -> int:
        """True length of the dataset (not limited).

        Returns:
            int: the full length
        """
        return self.dataset.full_length

    @property
    def inputs(self) -> Tensor:
        """Raw data of the dataset.

        Returns:
            Tensor: All of the data if available/possible, otherwise part of it.
        """
        return self.dataset.inputs[self.indices]

    @property
    def targets(self) -> Tensor:
        """Raw targets of the dataset.

        Returns:
            Tensor: All are part of the desired outputs.
        """
        return self.dataset.targets[self.indices]

    @property
    def num_classes(self) -> int:
        """Return the original number of classes in the data set.

        Returns:
            int: the number of classes.
        """
        return self.dataset.num_classes()


class DataLoader(TorchDataLoader, Generic[DatasetClass]):
    """Subclasses torch.utils.data.DataLoader for typing purposes."""

    dataset: DatasetClass


class BaseDataModule(
    LightningDataModule, Base, Generic[DatasetClass], metaclass=ABCMeta
):
    """Abstract class for all data modules."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        batch_size: int,
        dataset_class: Type[DatasetClass],
        dataset_limit: float = 1.0,
        num_workers: int = 0,
        resize: Optional[Tuple[int, int, int]] = None,
        dataset_dir: str = "data",
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.dataset_limit = dataset_limit
        self.num_workers = num_workers
        self.resize = resize
        self.dataset_dir = dataset_dir
        self.dataset_class = dataset_class

        self.train_set: Union[DatasetClass, Subset[DatasetClass]]
        self.val_set: Union[DatasetClass, Subset[DatasetClass]]

    @abstractmethod
    def setup(self, stage: Optional[str] = None) -> None:
        pass

    @property
    def num_classes(self) -> int:
        """Return the original number of classes of the underlying data set.

        Returns:
            int: the number of classes.
        """
        return self.dataset_class.num_classes()

    def target_transform(self) -> Optional[transforms.Compose]:
        """Optional transformations to be done on the targets before use."""
        return None

    def transform(self) -> Optional[transforms.Compose]:
        """Optional transformations to be done on the inputs before use."""
        return None

    def train_dataloader(self) -> DataLoader[DatasetClass]:
        """Returns the training dataloader.

        Returns:
            DataLoader[DatasetClass]: the initialized training data loader.
        """
        return DataLoader[DatasetClass](
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            shuffle=True,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        """Returns the validation dataloader.

        Returns:
            DataLoader[DatasetClass]: the initialized validation data loader.
        """
        return DataLoader[DatasetClass](
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            shuffle=False,
        )

    @property
    def mean(self) -> np.ndarray:
        """Return the mean of the dataset underlying dataset.

        Can be the mean over the whole dataset, the mean of each class, and even
        the mean per channel (or per class and per channel).
        TODO the output should be of shape (C), with C the number of channels.

        Returns:
            Tensor: the mean(s).
        """
        return self.dataset_class.mean()

    @property
    def std(self) -> np.ndarray:
        """Return the standard deviation of the underlying dataset.

        Can be the std over the whole dataset, the std of each class, and even
        the std per channel (or per class and per channel).
        TODO the output should be of shape (C), with C the number of channels.

        Returns:
            Tensor: the standard deviation(s).
        """
        return self.dataset_class.std()

    @property
    def sample(self) -> Tuple[Tensor, Tensor]:
        """Return a sample of the validation dataset.

        Returns:
            Tuple[Tensor, Tensor]: 10 samples of the validation data set and
                its targets.
        """
        return self.val_set.inputs[:10], self.val_set.targets[:10]
    
    
    @property
    def sample_transformed(self) -> Tuple[Tensor, Tensor]:
        """Return a sample of the validation dataset.

        Returns:
            Tuple[Tensor, Tensor]: 10 samples of the validation data set and
                its targets.
        """
        inputs, targets = next(iter(self.val_dataloader()))
        return inputs[:min(10, inputs.shape[0])], targets[:min(10, inputs.shape[0])],
    
    @property
    def steps_per_epoch(self) -> int:
        if not hasattr(self, "train_set"):
            raise AttributeError("`setup()` method needs to be called beforehand.")
        
        return math.ceil(len(self.train_set) / self.batch_size)


class BaseOneClassDataModule(BaseDataModule[DatasetClass], metaclass=ABCMeta):
    """Data module for one class classification"""

    def __init__(
        self,
        normal_class: int = 0,
        require_train_stat: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.normal_class = normal_class
        self.require_train_stat = require_train_stat

        self.outlier_classes = list(range(0, self.dataset_class.num_classes()))
        self.outlier_classes.remove(self.normal_class)

    @property
    def num_classes(self) -> int:
        """Returns the number of classes, which is 2 (normal, anomaly).

        Returns:
            int: the number of classes.
        """
        return 2  # 1: normal, 0: outlier

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        """Returns the validation dataloader(s).

        If some metrics are required from the normal class (train set), e.g. a
        threshold calculation, then returns 2 data loaders with the first one
        being on the train dataset.

        Returns:
            Union[DataLoader, List[DataLoader]]: the validation dataloader(s)
        """
        val_dataloader = DataLoader[DatasetClass](
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            shuffle=False,
        )

        if not self.require_train_stat:
            return val_dataloader

        return [
            DataLoader[DatasetClass](
                self.train_set,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=False,
                shuffle=False,
            ),
            val_dataloader,
        ]


class BasePartialDataModule(BaseDataModule[DatasetClass], metaclass=ABCMeta):
    """Data module over a dataset from which some classes are removed."""

    def __init__(
        self,
        excluded_classes: List[int],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.excluded_classes = excluded_classes
        self.remaining_classes = list(
            range(0, self.dataset_class.num_classes())
        )
        self.remaining_classes = list(
            set(self.remaining_classes) - set(self.excluded_classes)
        )

    @property
    def num_classes(self) -> int:
        """Return the remaining number of classes of the underlying data set.

        Returns:
            int: the number of classes.
        """
        return len(self.remaining_classes)


class BaseMultiNormalDataModule(BaseDataModule[DatasetClass], metaclass=ABCMeta):
    """Data module for one class classification"""

    def __init__(
        self,
        normal_classes: List[int],
        require_train_stat: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.normal_classes = normal_classes
        self.require_train_stat = require_train_stat

        self.outlier_classes = list(range(0, self.dataset_class.num_classes()))
        self.outlier_classes = list(
            set(self.outlier_classes) - set(self.normal_classes)
        )

    @property
    def num_classes(self) -> int:
        """Returns the number of classes, which is 2 (normal, anomaly).

        Returns:
            int: the number of classes.
        """
        return 2  # 1: normal, 0: outlier
