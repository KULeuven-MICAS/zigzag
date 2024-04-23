from abc import ABCMeta
from networkx import DiGraph


class Workload(DiGraph, metaclass=ABCMeta):
    """! Abstract Base Class for workloads"""

    pass
