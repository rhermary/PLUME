"""CIFAR10 dataloaders implementation."""

from typing import Optional, Tuple, Any, Dict

from torchvision.transforms import transforms
import torchvision
from torch import Tensor
import torch
import numpy as np
from kornia.color import rgb_to_hls
from torchvision.transforms.transforms import Compose

from .utils import OneHot, get_target_label_idx
from .base import (
    BaseDataModule,
    BaseDataset,
    BasePartialDataModule,
    Subset,
    BaseOneClassDataModule,
    BaseMultiNormalDataModule,
)


class CIFAR10Dataset(BaseDataset):
    """CIFAR10 data set."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        limit: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(limit=limit)

        self.dataset = torchvision.datasets.CIFAR10(**kwargs)

    @property
    def full_length(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.dataset[idx]

    @property
    def inputs(self) -> Tensor:
        return torch.from_numpy(self.dataset.data[: len(self)])

    @property
    def targets(self) -> Tensor:
        return torch.tensor(self.dataset.targets[: len(self)])

    @classmethod
    def num_classes(cls) -> int:
        return 10
    
    @classmethod
    def original_shape(cls) -> Tuple[int, int, int]:
        return (3, 32, 32)

    @classmethod
    def prepare_data(
        cls, *args: Any, dataset_dir: str = "", **kwargs: Any
    ) -> None:
        torchvision.datasets.CIFAR10(dataset_dir, train=True, download=True)
        torchvision.datasets.CIFAR10(dataset_dir, train=False, download=True)

    @classmethod
    def mean(cls) -> np.ndarray:
        # https://www.kaggle.com/code/fanbyprinciple/cifar10-explanation-with-pytorch
        # return np.array([0.4914, 0.4822, 0.4465])
        return np.array([
            [0.5256554097732844, 0.5603293658088235, 0.5889068098958333],
            [0.47118413449754903, 0.4545294178921569, 0.447198645067402],
            [0.4892499142156863, 0.49147702742034316, 0.4240447817095588],
            [0.49548247012867647, 0.4564121124387255, 0.4155386358762255],
            [0.47159063419117647, 0.4652057314644608, 0.3782071515012255],
            [0.4999258938419117, 0.4646367578125, 0.41654605085784313],
            [0.47005706035539213, 0.4383936764705882, 0.34521907245710787],
            [0.5019583601409313, 0.479863846507353, 0.4168859995404412],
            [0.49022592524509806, 0.5253946185661764, 0.5546856449142158],
            [0.4986669837622549, 0.48534152956495097, 0.4780763526348039]
        ])

    @classmethod
    def std(cls) -> np.ndarray:
        # return np.array([0.2470, 0.2435, 0.2616])
        return np.array([
            [0.25022019231571185, 0.2408348427129651, 0.2659734761252281],
            [0.26806353264459026, 0.2658273856706841, 0.2749459200708571],
            [0.2270547824974553, 0.22094457725751065, 0.2433792514147017],
            [0.25684312639087054, 0.2522707774794101, 0.25799371686534633],
            [0.21732735231631337, 0.20652700336972213, 0.2118233405487209],
            [0.2504253200447454, 0.24374875790308145, 0.24894635750868108],
            [0.22888339365158972, 0.21856169153937327, 0.22041993680516683],
            [0.2430489773244697, 0.24397302190495562, 0.251715596482951],
            [0.24962469788031316, 0.24068881282532456, 0.25149759373115593],
            [0.268052523912086, 0.2691079747712417, 0.2810165261230675]
        ])


class CIFAR10(BaseDataModule[CIFAR10Dataset]):
    """Classic CIFAR10 dataset, with one-hot encoded targets."""

    def __init__(
        self,
        force_grayscale: bool = False,
        **kwargs: Any,
    ) -> None:
        dataset_class = kwargs.pop("dataset_class", CIFAR10Dataset)
        super().__init__(dataset_class=dataset_class, **kwargs)

        self.force_grayscale = force_grayscale

    @property
    def mean(self) -> np.ndarray:
        return np.mean(self.dataset_class.mean(), axis=0)

    @property
    def std(self) -> np.ndarray:
        return np.mean(self.dataset_class.std(), axis=0)

    def prepare_data(self) -> None:
        self.dataset_class.prepare_data(dataset_dir=self.dataset_dir)

    def transform(self) -> transforms.Compose:
        transforms_ = [
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ]

        if self.resize is not None:
            if self.resize[0] == 1 or self.force_grayscale:
                transforms_.append(
                    transforms.Lambda(lambda x: rgb_to_hls(x)[1:2, :, :])
                )

            transforms_.append(transforms.Resize(self.resize[1:]))

            if self.force_grayscale:
                transforms_.append(
                    transforms.Lambda(lambda x: x.expand(*self.resize))
                )

        return transforms.Compose(transforms_)

    def target_transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Lambda(torch.tensor),
                OneHot(),
                transforms.Lambda(lambda x: x.to(torch.float32)),
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

        self.train_set = self.dataset_class(
            train=True,
            **kwargs,
        )

        self.val_set = self.dataset_class(
            train=False,
            **kwargs,
        )


class PartialCIFAR10(CIFAR10, BasePartialDataModule):
    """CIFAR10 with removable classes.

    TODO change mean() and std() if changed by class
    """

    def _subset(self, dataset: CIFAR10Dataset) -> Subset[CIFAR10Dataset]:
        """Subset train_set to only the normal class."""
        sub_idx = get_target_label_idx(dataset.targets, self.remaining_classes)
        return Subset(dataset, sub_idx)

    def target_transform(self) -> Optional[transforms.Compose]:
        return transforms.Compose(
            [
                super().target_transform(),
                transforms.Lambda(
                    lambda x: x[self.remaining_classes],
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
        self.val_set = self._subset(
            self.dataset_class(
                train=False,
                **kwargs,
            )
        )


class OneClassCIFAR10(CIFAR10, BaseOneClassDataModule):
    """CIFAR10 for one class classification.

    TODO change mean() and std() if changed by class
    """

    @property
    def mean(self) -> np.ndarray:
        return self.dataset_class.mean()[self.normal_class]

    @property
    def std(self) -> np.ndarray:
        return self.dataset_class.std()[self.normal_class]

    def transform(self) -> transforms.Compose:
        transforms_ = [
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ]

        if self.resize is not None:
            if self.resize[0] == 1 or self.force_grayscale:
                transforms_.append(
                    transforms.Lambda(lambda x: rgb_to_hls(x)[1:2, :, :])
                )

            transforms_.append(transforms.Resize(self.resize[1:]))

            if self.force_grayscale:
                transforms_.append(
                    transforms.Lambda(lambda x: x.expand(*self.resize))
                )

        return transforms.Compose(transforms_)
    
    def target_transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Lambda(
                    lambda x: torch.tensor(x not in self.outlier_classes).to(
                        torch.float32
                    )
                ),
            ]
        )

    def _subset(self, dataset: CIFAR10Dataset) -> Subset[CIFAR10Dataset]:
        """Subset train_set to only the normal class."""
        sub_idx = get_target_label_idx(dataset.targets, [self.normal_class])

        return Subset(dataset, sub_idx)

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


class JigsawCIFAR10Dataset(CIFAR10Dataset):
    def __init__(
        self,
        seed: int,
        permutations_path: str,
        transform: Optional[transforms.Compose] = None,
        target_transform: Optional[transforms.Compose] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.transform = transform
        self.target_transform = target_transform
        self.transform_generator = torch.Generator().manual_seed(seed)

        # https://github.com/bbrattoli/JigsawPuzzlePytorch/blob/master/permutations_1000.npy
        self.permutations = torch.from_numpy(
            np.load(f"{self.dataset.root}/permutations_1000.npy")
        )
        
        self.kernel_size = self.original_shape()[-1] // 3 # Assumed square
        self.unfolder = torch.nn.Unfold(
            kernel_size=self.kernel_size, stride=self.kernel_size
        )
        self.refolder = torch.nn.Fold(
            output_size=(self.kernel_size * 3, self.kernel_size * 3),
            kernel_size=self.kernel_size,
            stride=self.kernel_size
        )

    @classmethod
    def num_classes(cls) -> int:
        return 1000
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        permutation_idx = torch.randint(
            low=0,
            high=self.num_classes(),
            size=(1,),
            generator=self.transform_generator
        )[0]

        input_, _ = super().__getitem__(idx)
        input_ = transforms.ToTensor()(input_)
        target = permutation_idx
        
        permutation = self.permutations[permutation_idx]

        permuted = self.unfolder(input_[None, :, :, :])[0, :, permutation]
        reconstructed = self.refolder(permuted)

        if self.transform:
            reconstructed = self.transform(reconstructed)

        if self.target_transform:
            target = self.target_transform(target)

        return reconstructed, target


class JigsawCIFAR10(CIFAR10):  # Dataset class here?
    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        dataset_class = kwargs.pop("dataset_class", JigsawCIFAR10Dataset)
        super().__init__(dataset_class=dataset_class, **kwargs)

        self.seed_ = torch.random.seed()
        self.log_param("random_seed", self.seed_)
        if self.resize is None:
            self.resize = self.dataset_class.original_shape()


    def transform(self) -> transforms.Compose:
        transforms_ = [
            transforms.Normalize(mean=self.mean, std=self.std),
        ]

        if self.resize[0] == 1 or self.force_grayscale:
            transforms_.append(
                transforms.Lambda(lambda x: rgb_to_hls(x)[1:2, :, :])
            )

        transforms_.append(transforms.Resize(self.resize[1:]))

        if self.force_grayscale:
            transforms_.append(
                transforms.Lambda(lambda x: x.expand(*self.resize))
            )

        return transforms.Compose(transforms_)

    def target_transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                OneHot(num_classes=self.dataset_class.num_classes()),
                transforms.Lambda(lambda x: x.to(torch.float32)),
            ]
        )

    def _get_dataset_args(self) -> Dict[str, Any]:
        """Common arguments to the train and validation datasets."""
        return {
            "root": self.dataset_dir,
            "transform": self.transform(),
            "target_transform": self.target_transform(),
            "limit": self.dataset_limit,
            "permutations_path": "",  # TODO make it an argument
        }

    def setup(self, stage: Optional[str] = None) -> None:
        kwargs = self._get_dataset_args()

        self.train_set = self.dataset_class(
            train=True,
            seed=self.seed_,
            **kwargs,
        )

        self.val_set = self.dataset_class(
            train=False,
            seed=42,
            **kwargs,
        )


class PartialJigsawCIFAR10(JigsawCIFAR10, BasePartialDataModule):
    def _subset(self, dataset: JigsawCIFAR10Dataset) -> Subset[JigsawCIFAR10Dataset]:
        sub_idx = get_target_label_idx(dataset.targets, self.remaining_classes)
        return Subset(dataset, sub_idx)

    def target_transform(self) -> Optional[transforms.Compose]:
        return transforms.Compose(
            [
                super().target_transform(),
                transforms.Lambda(
                    lambda x: x[self.remaining_classes],
                ),
            ]
        )

    def setup(self, stage: Optional[str] = None) -> None:
        kwargs = self._get_dataset_args()

        self.train_set = self._subset(
            self.dataset_class(
                train=True,
                seed=self.seed_,
                **kwargs,
            )
        )
        self.val_set = self._subset(
            self.dataset_class(
                train=False,
                seed=42,
                **kwargs,
            )
        )

class MultiNormalCIFAR10(CIFAR10, BaseMultiNormalDataModule):
    """CIFAR10 with multiple normal classes.

    TODO change mean() and std() if changed by class
    """

    def _subset(self, dataset: CIFAR10Dataset) -> Subset[CIFAR10Dataset]:
        """Subset train_set to only the normal class."""
        sub_idx = get_target_label_idx(dataset.targets, self.normal_classes)
        return Subset(dataset, sub_idx)

    def target_transform(self) -> Optional[transforms.Compose]:
        return transforms.Compose(
            [
                transforms.Lambda(
                    lambda x: torch.tensor(x not in self.outlier_classes).to(
                        torch.float32
                    )
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
