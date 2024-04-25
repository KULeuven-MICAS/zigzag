from abc import ABCMeta, abstractmethod
from typing import Iterator
from zigzag.utils import json_repr_handler


class LayerAttribute(metaclass=ABCMeta):
    """! Abstract Base Class to represent any layer attribute"""

    @abstractmethod
    def __init__(self, data):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator:
        return iter(self.data)

    def __getitem__(self, key):
        # TODO: this is dangerous for some subclasses
        return self.data[key]

    def __contains__(self, key) -> bool:
        return key in self.data

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return str(self.data)

    def __jsonrepr__(self):
        return json_repr_handler(self.data)

    @staticmethod
    @abstractmethod
    def parse_user_input(x): ...
