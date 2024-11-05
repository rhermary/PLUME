from typing import Tuple, Any, Optional, Dict
import os

import torch
from torch import Tensor
from torchvision import transforms
import numpy as np
from numpy.lib import npyio

from .base import BaseDataset, BaseDataModule, BaseOneClassDataModule, Subset
from .utils import OneHot, get_target_label_idx


class SPARKDataset(BaseDataset):
    NUM_IMAGES_PER_FILE = 1000
    NUM_TRAIN_FILES = 90
    NUM_VAL_FILES = 30
    ZFILL = 5
    BASE_FOLDER = "SPARK-P2"

    def __init__(
        self,
        limit: float = 1,
        transform: Optional[transforms.Compose] = None,
        target_transform: Optional[transforms.Compose] = None,
        train: bool = False,
        root: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(limit=limit)

        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.root = (
            f"{root}/{self.BASE_FOLDER}/{'train' if self.train else 'val'}"
        )

        self.load_data()

        # self.inputs_ : Dict[int, npyio.NpzFile]
        self.targets_: np.ndarray

    def load_data(self) -> None:
        # Inputs ZIP files (.npz) cannot be preloaded if using multiple workers;
        # issues arise and the training fails. Everything needs to be done at
        # runtime in that case, which seems not to have a big impact anyway as
        # NpzFile are just "meta files" and arrays are loaded when accessing
        # them.
        # https://github.com/numpy/numpy/issues/18124
        #
        # nb_files = self.NUM_TRAIN_FILES if self.train else self.NUM_VAL_FILES
        # self.inputs_ = dict()

        # for idx in range(nb_files):
        #     self.inputs_[idx] = np.load(f"{self.root}/{str(idx).zfill(self.ZFILL)}.npz")

        self.targets_ = np.load(f"{self.root}/targets.npy")

    @classmethod
    def num_classes(cls) -> int:
        return 11
    
    @classmethod
    def original_shape(cls) -> Tuple[int, int, int]:
        return (3, 32, 32)

    @property
    def full_length(self) -> int:
        return len(self.targets_)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs: npyio.NpzFile = np.load(
            f"{self.root}/{str(idx // self.NUM_IMAGES_PER_FILE).zfill(self.ZFILL)}.npz"
        )

        input_ = getattr(inputs.f, f"arr_{idx % self.NUM_IMAGES_PER_FILE}")
        target = int(self.targets_[idx])

        if self.transform is not None:
            input_ = self.transform(input_)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return input_, target  # type: ignore

    @property
    def inputs(self) -> Tensor:
        # Needs a bypass not to have normalized samples, which would not be
        # consistent with the other data sets.
        # Only returns the first 100 samples (or less if a lower limit was set).
        transforms_ = self.transform, self.target_transform
        self.transform, self.target_transform = None, None

        inputs_ = np.concatenate(
            [self[idx][0][None, :, :, :] for idx in range(min(len(self), 100))]
        )

        self.transform, self.target_transform = transforms_
        return torch.from_numpy(inputs_)

    @property
    def targets(self) -> Tensor:
        return torch.from_numpy(self.targets_[: len(self)])

    @classmethod
    def prepare_data(
        cls, *args: Any, dataset_dir: str = "", **kwargs: Any
    ) -> None:
        assert os.path.isdir(f"{dataset_dir}/{cls.BASE_FOLDER}/train")
        assert os.path.isdir(f"{dataset_dir}/{cls.BASE_FOLDER}/val")
        assert (
            len(os.listdir(f"{dataset_dir}/{cls.BASE_FOLDER}/train"))
            == cls.NUM_TRAIN_FILES + 1
        )
        assert (
            len(os.listdir(f"{dataset_dir}/{cls.BASE_FOLDER}/val"))
            == cls.NUM_VAL_FILES + 1
        )

    @classmethod
    def mean(cls) -> np.ndarray:
        if "P2" in cls.BASE_FOLDER:
            return np.array([0.09430549, 0.09299065, 0.09526419])
        
        return np.array([0.09244801, 0.09198833, 0.60900891])

    @classmethod
    def std(cls) -> np.ndarray:
        if "P2" in cls.BASE_FOLDER:
            return np.array([0.09868433, 0.09484387, 0.09236576])
        
        return np.array([0.10071791, 0.09504412, 0.33186111])

class SPARK(BaseDataModule[SPARKDataset]):
    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        dataset_class = kwargs.pop("dataset_class", SPARKDataset)
        super().__init__(dataset_class=dataset_class, **kwargs)

        self.satellite_classes = list(set(range(0, 11)) - set({6}))

    def prepare_data(self) -> None:
        self.dataset_class.prepare_data(dataset_dir=self.dataset_dir)

    def transform(self) -> Optional[transforms.Compose]:
        transforms_ = [
            transforms.Lambda(
                lambda x: torch.tensor(x, dtype=torch.float) / 255
            ),
            transforms.Normalize(mean=self.mean, std=self.std),
        ]
    
        if self.resize is not None:
            transforms_.append(transforms.Resize(self.resize[1:]))

        return transforms.Compose(transforms_)

    def _subset(self, dataset: SPARKDataset) -> Subset[SPARKDataset]:
        """Subset train_set to only the satellites classes."""
        sub_idx = get_target_label_idx(dataset.targets, self.satellite_classes)
        return Subset(dataset, sub_idx)

    def target_transform(self) -> Optional[transforms.Compose]:
        return transforms.Compose(
            [
                transforms.Lambda(torch.tensor),
                OneHot(self.dataset_class.num_classes()),
                transforms.Lambda(lambda x: x.to(torch.float32)),
                transforms.Lambda(
                    lambda x: x[self.satellite_classes],
                ),
            ]
        )

    def _get_dataset_args(self) -> Dict[str, Any]:
        """Common arguments to the train and validation datasets."""
        return {
            "root": self.dataset_dir,
            "transform": self.transform(),
            "target_transform": self.target_transform(),
            "limit": self.dataset_limit,
        }

    def setup(self, stage: Optional[str] = None) -> None:
        kwargs = self._get_dataset_args()

        self.train_set = self._subset(
            self.dataset_class(
                train=True,
                **kwargs,
            )
        )

        self.val_set = self._subset(
            self.dataset_class(
                train=False,
                **kwargs,
            )
        )

    @property
    def num_classes(self) -> int:
        return self.dataset_class.num_classes() - 1


class OneClassSPARK(SPARK, BaseOneClassDataModule):
    def target_transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Lambda(
                    lambda x: torch.tensor(x != 6).to(torch.float32)
                ),
            ]
        )

    def setup(self, stage: Optional[str] = None) -> None:
        kwargs = self._get_dataset_args()

        self.train_set = self._subset(
            self.dataset_class(
                train=True,
                **kwargs,
            )
        )

        self.val_set = self.dataset_class(
            train=False,
            **kwargs,
        )
