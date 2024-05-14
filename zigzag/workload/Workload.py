import networkx as nx
from abc import ABCMeta
from typing import Any, Iterator, Sequence
from networkx import DiGraph

from zigzag.workload.DummyNode import DummyNode
from zigzag.workload.layer_node import LayerNode


class Workload(DiGraph, metaclass=ABCMeta):
    """! Abstract Base Class for workloads"""

    def __init__(self, **attr: Any):
        super().__init__(**attr)  # type: ignore

    def topological_sort(self) -> Iterator[LayerNode | DummyNode]:
        return nx.topological_sort(self)  # type: ignore

    def add_workload_node(self, node: LayerNode | DummyNode) -> None:
        self.add_node(node)  # type: ignore

    def add_workload_edges_from(self, edges: Sequence[tuple[LayerNode | DummyNode, LayerNode | DummyNode]]) -> None:
        self.add_edges_from(edges)  # type: ignore

    def get_node_with_id(self, node_id: int) -> LayerNode | DummyNode:
        for node in self.nodes:  # type: ignore
            if node.id == node_id:  # type: ignore
                return node  # type: ignore
        raise ValueError(f"Node with id {node_id} not found in workload")
