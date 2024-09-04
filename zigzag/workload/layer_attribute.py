from abc import ABCMeta, abstractmethod
from typing import Any, Iterator

from zigzag.utils import json_repr_handler


class LayerAttribute(metaclass=ABCMeta):
    """! Abstract Base Class to represent any layer attribute"""

    @abstractmethod
    def __init__(self, data: Any):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.data)

    def __getitem__(self, key: Any):
        return self.data[key]

    def __contains__(self, key: Any) -> bool:
        return key in self.data

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return str(self.data)

    def __jsonrepr__(self) -> Any:
        return json_repr_handler(self.data)

    def __eq__(self, other: object):
        return isinstance(other, LayerAttribute) and self.data == other.data
