from abc import ABCMeta
from typing import Any
from networkx import DiGraph


class Workload(DiGraph, metaclass=ABCMeta):
    """! Abstract Base Class for workloads"""

    pass

    # def __init__(self, **attr: Any):
    #     super().__init__(**attr)
