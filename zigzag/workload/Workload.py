import networkx as nx
from abc import ABCMeta
from typing import Any, Iterator, Sequence
from networkx import DiGraph

from zigzag.workload.LayerNodeABC import LayerNodeABC


class Workload(DiGraph, metaclass=ABCMeta):
    """! Abstract Base Class for workloads"""

    def __init__(self, **attr: Any):
        super().__init__(**attr)  # type: ignore

    def topological_sort(self) -> Iterator[LayerNodeABC]:
        return nx.topological_sort(self)  # type: ignore

    def add_workload_node(self, node: LayerNodeABC) -> None:
        self.add_node(node)  # type: ignore

    def add_workload_edges_from(self, edges: Sequence[tuple[LayerNodeABC, LayerNodeABC]]) -> None:
        self.add_edges_from(edges)  # type: ignore

    def get_node_with_id(self, node_id: int) -> LayerNodeABC:
        for node in self.node_iterator:
            if node.id == node_id:
                return node
        raise ValueError(f"Node with id {node_id} not found in workload")

    @property
    def node_iterator(self) -> Iterator[LayerNodeABC]:
        return self.nodes()
