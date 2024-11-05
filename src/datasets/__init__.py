"""Modules for the implemented datasets."""

from .cifar10 import *
from .cifar100 import *
from .features import *
from .spark import *

from .base import BaseDataModule, BaseOneClassDataModule, BasePartialDataModule, BaseMultiNormalDataModule

DATASETS = BaseDataModule.listing()
ONE_CLASS_DATASETS = BaseOneClassDataModule.listing()
PARTIAL_DATASETS = BasePartialDataModule.listing()
MULTINORMAL_DATASETS = BaseMultiNormalDataModule.listing()
