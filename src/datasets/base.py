from abc import ABC, abstractmethod


class BaseDataset(ABC):
    @abstractmethod
    def __init__(self, path):
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, key: int | slice) -> dict | list[dict]:
        ...

    @abstractmethod
    def load(self) -> list[dict]:
        ...
