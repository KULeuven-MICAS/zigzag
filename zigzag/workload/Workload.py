from abc import ABCMeta
from typing import Any, Generic, Iterator, Sequence, TypeVar

import networkx as nx
from networkx import DiGraph

from zigzag.workload.layer_node import LayerNode
from zigzag.workload.LayerNodeABC import LayerNodeABC

T = TypeVar("T", bound=LayerNodeABC)


class WorkloadABC(DiGraph, Generic[T], metaclass=ABCMeta):
    """! Abstract Base Class for workloads, parameterizable with type T, which must be a (subclass from) LayerNodeABC"""

    def __init__(self, **attr: Any):
        super().__init__(**attr)  # type: ignore

    def topological_sort(self) -> Iterator[T]:
        return nx.topological_sort(self)  # type: ignore

    def add_workload_node(self, node: T) -> None:
        self.add_node(node)  # type: ignore

    def remove_workload_nodes_from(self, nodes: Iterator[T]) -> None:
        self.remove_nodes_from(nodes)  # type: ignore

    def add_workload_edge(self, edge_from: T, edge_to: T) -> None:
        self.add_edge(edge_from, edge_to)  # type: ignore

    def add_workload_edges_from(self, edges: Sequence[tuple[T, T]]) -> None:
        self.add_edges_from(edges)  # type: ignore

    def get_out_degree_for_layer(self, layer: T) -> int:
        return self.out_degree(layer)  # type: ignore

    def get_in_degree_for_layer(self, layer: T) -> int:
        return self.in_degree(layer)  # type: ignore

    def get_successors_for_layer(self, layer: T) -> Iterator[T]:
        return self.successors(layer)  # type: ignore

    def get_predecessors_for_layer(self, layer: T) -> Iterator[T]:
        return self.predecessors(layer)  # type: ignore

    def get_node_with_id(self, node_id: int) -> T:
        for node in self.node_list:
            if node.id == node_id:
                return node
        raise ValueError(f"Node with id {node_id} not found in workload")

    def get_copy_no_dummy(self) -> "WorkloadABC[LayerNode]":
        ...

    @property
    def node_list(self) -> list[T]:
        return list(self.nodes())  # type: ignore
