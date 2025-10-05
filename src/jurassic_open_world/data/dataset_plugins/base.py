from abc import ABC, abstractmethod

from torch.utils.data import Dataset

from ...utils.types import DatasetPluginOutput


class BaseDatasetPlugin(ABC, Dataset):
    @abstractmethod
    def __getitem__(self, index: int) -> DatasetPluginOutput: ...

    @abstractmethod
    def __len__(self) -> int: ...
