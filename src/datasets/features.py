"""Allowing datasets to be extracted features instead of raw images."""
from typing import Any, Tuple, Optional, Dict
from pathlib import Path

import torch
from torchvision import transforms

from .cifar10 import OneClassCIFAR10, CIFAR10Dataset, CIFAR10
from .cifar100 import OneClassCIFAR100, CIFAR100Dataset, CIFAR100
from .spark import SPARK, SPARKDataset, OneClassSPARK


class CIFAR10FeaturesDataset(CIFAR10Dataset):
    def __init__(
        self,
        limit: float = 1,
        features_dir: str = "",
        transform: Optional[transforms.Compose] = None,
        train: bool = False,
        root: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            limit, train=train, root=root, transform=transform, **kwargs
        )

        self.transform = transform

        del self.dataset.data
        self.dataset.data = self.load_data(Path(root) / features_dir, train)

    def load_data(self, dataset_dir: Path, train: bool = False) -> torch.Tensor:
        if train:
            path = (dataset_dir / "train.pt").as_posix()
        else:
            path = (dataset_dir / "val.pt").as_posix()

        return torch.load(path)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features, target = self.dataset.data[idx], self.dataset.targets[idx]

        if self.transform is not None:
            features = self.transform(features)

        if self.dataset.target_transform is not None:
            target = self.dataset.target_transform(target)

        return features, target

    @classmethod
    def original_shape(cls) -> Tuple[int, int, int]:
        return (1, 1, 3072)
    
    @classmethod
    def prepare_data(
        cls, *args: Any, dataset_dir: str = "", features_dir: str = "", **kwargs: Any
    ) -> None:
        CIFAR10Dataset.prepare_data(dataset_dir=dataset_dir)
        assert (Path(dataset_dir) / features_dir / "val.pt").is_file()
        assert (Path(dataset_dir) / features_dir / "train.pt").is_file()


class CIFAR10Features(CIFAR10):
    def __init__(self, features_dir: str = "", **kwargs: Any) -> None:
        super().__init__(dataset_class=CIFAR10FeaturesDataset, **kwargs)

        self.features_dir = features_dir

    def prepare_data(self) -> None:
        self.dataset_class.prepare_data(dataset_dir=self.dataset_dir, features_dir=self.features_dir)

    def _get_dataset_args(self) -> Dict[str, Any]:
        return super()._get_dataset_args() | {"features_dir": self.features_dir}

    def transform(self) -> Optional[transforms.Compose]:
        if self.resize is None:
            return None
        
        return transforms.Compose(
            [
                transforms.Lambda(lambda x: x.reshape(3, 32, 32)),
            ]
        )


class OneClassCIFAR10Features(OneClassCIFAR10):
    def __init__(self, features_dir: str = "", **kwargs: Any) -> None:
        super().__init__(dataset_class=CIFAR10FeaturesDataset, **kwargs)

        self.features_dir = features_dir

    def prepare_data(self) -> None:
        self.dataset_class.prepare_data(dataset_dir=self.dataset_dir, features_dir=self.features_dir)

    def _get_dataset_args(self) -> Dict[str, Any]:
        return super()._get_dataset_args() | {"features_dir": self.features_dir}

    def transform(self) -> Optional[transforms.Compose]:
        if self.resize is None:
            return None
        
        return transforms.Compose(
            [
                transforms.Lambda(lambda x: x.reshape(3, 32, 32)),
            ]
        )


class CIFAR100FeaturesDataset(CIFAR100Dataset):
    def __init__(
        self,
        limit: float = 1,
        features_dir: str = "",
        transform: Optional[transforms.Compose] = None,
        train: bool = False,
        root: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            limit, train=train, root=root, transform=transform, **kwargs
        )

        self.transform = transform

        del self.dataset.data
        self.dataset.data = self.load_data(Path(root) / features_dir, train)

    def load_data(self, dataset_dir: Path, train: bool = False) -> torch.Tensor:
        if train:
            path = (dataset_dir / "train.pt").as_posix()
        else:
            path = (dataset_dir / "val.pt").as_posix()

        return torch.load(path)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features, target = self.dataset.data[idx], self.dataset.targets[idx]

        if self.transform is not None:
            features = self.transform(features)

        if self.dataset.target_transform is not None:
            target = self.dataset.target_transform(target)

        return features, target

    @classmethod
    def original_shape(cls) -> Tuple[int, int, int]:
        return (1, 1, 3072)
    
    @classmethod
    def prepare_data(
        cls, *args: Any, dataset_dir: str = "", features_dir: str = "", **kwargs: Any
    ) -> None:
        CIFAR100Dataset.prepare_data(dataset_dir=dataset_dir)
        assert (Path(dataset_dir) / features_dir / "val.pt").is_file()
        assert (Path(dataset_dir) / features_dir / "train.pt").is_file()


class CIFAR100Features(CIFAR100):
    def __init__(self, features_dir: str = "", **kwargs: Any) -> None:
        super().__init__(dataset_class=CIFAR100FeaturesDataset, **kwargs)

        self.features_dir = features_dir

    def prepare_data(self) -> None:
        self.dataset_class.prepare_data(dataset_dir=self.dataset_dir, features_dir=self.features_dir)

    def _get_dataset_args(self) -> Dict[str, Any]:
        return super()._get_dataset_args() | {"features_dir": self.features_dir}

    def transform(self) -> Optional[transforms.Compose]:
        if self.resize is None:
            return None
        
        return transforms.Compose(
            [
                transforms.Lambda(lambda x: x.reshape(3, 32, 32)),
            ]
        )


class OneClassCIFAR100Features(OneClassCIFAR100):
    def __init__(self, features_dir: str = "", **kwargs: Any) -> None:
        super().__init__(dataset_class=CIFAR100FeaturesDataset, **kwargs)

        self.features_dir = features_dir

    def prepare_data(self) -> None:
        self.dataset_class.prepare_data(dataset_dir=self.dataset_dir, features_dir=self.features_dir)

    def _get_dataset_args(self) -> Dict[str, Any]:
        return super()._get_dataset_args() | {"features_dir": self.features_dir}

    def transform(self) -> Optional[transforms.Compose]:
        if self.resize is None:
            return None
        
        return transforms.Compose(
            [
                transforms.Lambda(lambda x: x.reshape(3, 32, 32)),
            ]
        )


class SPARKFeaturesDataset(SPARKDataset):
    def __init__(
        self,
        limit: float = 1,
        features_dir: str = "",
        transform: Optional[transforms.Compose] = None,
        train: bool = False,
        root: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            limit, train=train, root=root, transform=transform, **kwargs
        )

        self.inputs_ = self.load_data_(Path(root) / features_dir, train)

    def load_data_(
        self, dataset_dir: Path, train: bool = False
    ) -> torch.Tensor:
        if train:
            path = (dataset_dir / "train.pt").as_posix()
        else:
            path = (dataset_dir / "val.pt").as_posix()

        return torch.load(path)

    @classmethod
    def original_shape(cls) -> Tuple[int, int, int]:
        return (1, 1, 3072)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        target: Any
        features, target = self.inputs_[idx], int(self.targets_[idx])

        if self.transform is not None:
            features = self.transform(features)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return features, target

    @classmethod
    def prepare_data(
        cls, *args: Any, dataset_dir: str = "", features_dir: str = "", **kwargs: Any
    ) -> None:
        SPARKDataset.prepare_data(dataset_dir=dataset_dir)
        assert (Path(dataset_dir) / features_dir / "val.pt").is_file()
        assert (Path(dataset_dir) / features_dir / "train.pt").is_file()


class SPARKFeatures(SPARK):
    def __init__(self, features_dir: str = "", **kwargs: Any) -> None:
        super().__init__(dataset_class=SPARKFeaturesDataset, **kwargs)

        self.features_dir = features_dir

    def prepare_data(self) -> None:
        self.dataset_class.prepare_data(dataset_dir=self.dataset_dir, features_dir=self.features_dir)

    def _get_dataset_args(self) -> Dict[str, Any]:
        return super()._get_dataset_args() | {"features_dir": self.features_dir}

    def transform(self) -> Optional[transforms.Compose]:
        # TODO: will crash for old trainings without 3x32x32 output
        return transforms.Compose(
            [
                transforms.Lambda(lambda x: x.reshape(3, 32, 32)),
            ]
        )


class OneClassSPARKFeatures(OneClassSPARK):
    def __init__(self, features_dir: str = "", **kwargs: Any) -> None:
        super().__init__(dataset_class=SPARKFeaturesDataset, **kwargs)

        self.features_dir = features_dir

    def prepare_data(self) -> None:
        self.dataset_class.prepare_data(dataset_dir=self.dataset_dir, features_dir=self.features_dir)

    def _get_dataset_args(self) -> Dict[str, Any]:
        return super()._get_dataset_args() | {"features_dir": self.features_dir}

    def transform(self) -> Optional[transforms.Compose]:
        if self.resize is None:
            return None
        
        return transforms.Compose(
            [
                transforms.Lambda(lambda x: x.reshape(3, 32, 32)),
            ]
        )
