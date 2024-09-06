from abc import ABCMeta
from typing import TypeVar

from zigzag.utils import DiGraphWrapper
from zigzag.workload.layer_node import LayerNode
from zigzag.workload.layer_node_abc import LayerNodeABC

T = TypeVar("T", bound=LayerNodeABC)


class WorkloadABC(DiGraphWrapper[LayerNodeABC], metaclass=ABCMeta):
    """! Abstract Base Class for workloads, parameterizable with type T, which must be a (subclass from) LayerNodeABC"""

    def get_copy_no_dummy(self) -> "WorkloadNoDummyABC": ...


class WorkloadNoDummyABC(DiGraphWrapper[LayerNode], metaclass=ABCMeta):
    "Abstract bass class for workloads with only simulatable nodes"
