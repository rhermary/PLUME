"""CIFAR10 dataloaders implementation."""

from typing import Optional, Tuple, Any, Dict

from torchvision.transforms import transforms
import torchvision
from torch import Tensor
import torch
import numpy as np
from kornia.color import rgb_to_hls

from .utils import OneHot, get_target_label_idx
from .base import (
    BaseDataModule,
    BaseDataset,
    Subset,
    BaseOneClassDataModule,
)


class CIFAR100Dataset(BaseDataset):
    """CIFAR100 data set."""

    META_CLASSES = [
        [4, 30, 55, 72, 95],
        [1, 32, 67, 73, 91],
        [54, 62, 70, 82, 92],
        [9, 10, 16, 28, 61],
        [0, 51, 53, 57, 83],
        [22, 39, 40, 86, 87],
        [5, 20, 25, 84, 94],
        [6, 7, 14, 18, 24],
        [3, 42, 43, 88, 97],
        [12, 17, 37, 68, 76],
        [23, 33, 49, 60, 71],
        [15, 19, 21, 31, 38],
        [34, 63, 64, 66, 75],
        [26, 45, 77, 79, 99],
        [2, 11, 35, 46, 98],
        [27, 29, 44, 78, 93],
        [36, 50, 65, 74, 80],
        [47, 52, 56, 59, 96],
        [8, 13, 48, 58, 90],
        [41, 69, 81, 85, 89]
    ]

    def __init__(  # pylint: disable=too-many-arguments
        self,
        limit: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(limit=limit)

        self.dataset = torchvision.datasets.CIFAR100(**kwargs)

        self.meta_targets = list(map(self.meta_targets_mapping().__getitem__, self.dataset.targets))

    @classmethod
    def meta_targets_mapping(cls):
        return {
            y: idx
            for idx, x in enumerate(cls.META_CLASSES)
            for y in x
        }
    
    @property
    def full_length(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.dataset[idx]

    @property
    def inputs(self) -> Tensor:
        return torch.from_numpy(self.dataset.data)

    @property
    def targets(self) -> Tensor:
        return torch.tensor(self.meta_targets)

    @classmethod
    def num_classes(cls) -> int:
        return 20
    
    @classmethod
    def original_shape(cls) -> Tuple[int, int, int]:
        return (3, 32, 32)

    @classmethod
    def prepare_data(
        cls, *args: Any, dataset_dir: str = "", **kwargs: Any
    ) -> None:
        torchvision.datasets.CIFAR100(dataset_dir, train=True, download=True)
        torchvision.datasets.CIFAR100(dataset_dir, train=False, download=True)

    @classmethod
    def mean(cls) -> np.ndarray:
        return np.array([
            [0.42423144914119465, 0.47832111213175177, 0.4975257092518573],
            [0.4162160569838433, 0.46257881433736103, 0.4577342386627789],
            [0.5233250091914058, 0.4079774601703923, 0.30579033547521095],
            [0.5823847012900129, 0.5459058946090704, 0.5111448023893325],
            [0.601014928005143, 0.49265376684912787, 0.3616291835149617],
            [0.5573075582112769, 0.5368135355386912, 0.5250962806363704],
            [0.6103587622589342, 0.5407373299626262, 0.4902287285516677],
            [0.5568150597436623, 0.5380312285545601, 0.4311980346184618],
            [0.4719118060655543, 0.4394540058203162, 0.3713119194233891],
            [0.4847807429525375, 0.5094398054533787, 0.5115890379896408],
            [0.4684283318002898, 0.4970663786761851, 0.5026027251835199],
            [0.4832378921560621, 0.4649729917266515, 0.3996753844977545],
            [0.45113381586922763, 0.42913893688670013, 0.36685374999966264],
            [0.5092925658699786, 0.4759502726699119, 0.4163877466288309],
            [0.5192720802703117, 0.4569829687482902, 0.42417880361370225],
            [0.48772139399457337, 0.4780939537362393, 0.4335733440555291],
            [0.5116775842526773, 0.47981878370065734, 0.4124985983450676],
            [0.454443357842015, 0.4832513710159148, 0.4314423973637591],
            [0.49785809282986276, 0.48577209405495175, 0.46422195925059634],
            [0.5300919960164487, 0.528016761641328, 0.5036738878658933],
    ])

    @classmethod
    def std(cls) -> np.ndarray:
        return np.array([
            [0.24950120520414407, 0.23273790524829993, 0.25147213870598306],
            [0.28571281729985437, 0.2574223632236713, 0.2704952410447696],
            [0.28919808181158213, 0.2488546790384656, 0.26133859512205015],
            [0.2760762124262232, 0.27824263213493533, 0.28875053693402464],
            [0.29020673070610137, 0.28373780733068704, 0.29521147508854756],
            [0.29334023611304455, 0.29370714717402163, 0.3028157431157146],
            [0.2603200021101814, 0.2773577391233595, 0.2955911054940095],
            [0.273615064274395, 0.26302498231461513, 0.29230921165278284],
            [0.23524579697837913, 0.22442165257841223, 0.22762841461357872],
            [0.24081619613131017, 0.24106794521850217, 0.2736209334451734],
            [0.23492771036862944, 0.2283152286489116, 0.2730908258718698],
            [0.24279335268179558, 0.23754348009485796, 0.23727379401156196],
            [0.23550961320887864, 0.22482952314529106, 0.22607167077062096],
            [0.26448233113408426, 0.2566131539944638, 0.2683540512225859],
            [0.2800751220989466, 0.27352885440714547, 0.2793880550174855],
            [0.2618900573536653, 0.2417568053274001, 0.2526356515781338],
            [0.238881877105578, 0.22876998599962206, 0.23766514748540635],
            [0.23560031758642133, 0.23447846592609126, 0.28167326878304094],
            [0.26328007512937884, 0.2609428109345766, 0.2733392841179149],
            [0.27296178118092734, 0.2692785019593437, 0.28741607330860164],
        ])


class CIFAR100(BaseDataModule[CIFAR100Dataset]):
    def __init__(
        self,
        force_grayscale: bool = False,
        **kwargs: Any,
    ) -> None:
        dataset_class = kwargs.pop("dataset_class", CIFAR100Dataset)
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
                transforms.Lambda(CIFAR100Dataset.meta_targets_mapping().__getitem__),
                transforms.Lambda(torch.tensor),
                OneHot(self.dataset_class.num_classes()),
                transforms.Lambda(lambda x: x.to(torch.float32)),
            ]
        )

    def _get_dataset_args(self) -> Dict[str, Any]:
        """Common arguments to the train and validation datasets."""
        return {
            "root": self.dataset_dir,
            "transform": self.transform(),
            "target_transform": self.target_transform(),
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

class OneClassCIFAR100(CIFAR100, BaseOneClassDataModule):
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
                transforms.Lambda(CIFAR100Dataset.meta_targets_mapping().__getitem__),
                transforms.Lambda(
                    lambda x: torch.tensor(x not in self.outlier_classes).to(
                        torch.float32
                    )
                ),
            ]
        )

    def _subset(self, dataset: CIFAR100Dataset) -> Subset[CIFAR100Dataset]:
        """Subset train_set to only the normal class."""
        sub_idx = get_target_label_idx(dataset.meta_targets, [self.normal_class])

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
