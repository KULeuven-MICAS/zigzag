import logging
import pickle
from copy import deepcopy
from hashlib import sha512  # type: ignore
from typing import Any, Generic, Iterator, Literal, Sequence, TypeVar, no_type_check, overload

import networkx as nx
import numpy as np
import yaml
from networkx import DiGraph
from typeguard import typeguard_ignore  # type: ignore


def hash_sha512(data: Any) -> int:
    """! Hashes the input data using SHA-512"""
    return int(sha512(pickle.dumps(data)).hexdigest(), 16)  # type: ignore


def pickle_deepcopy(to_copy: Any) -> Any:
    try:
        copy = pickle.loads(pickle.dumps(to_copy, -1))
        return copy
    except:  # noqa: E722 # pylint: disable=W0702
        return deepcopy(to_copy)


def pickle_save(to_save: str, path: str):
    with open(path, "wb") as fp:
        status = pickle.dump(to_save, fp)
    return status


def pickle_load(path: str):
    with open(path, "rb") as fp:
        obj = pickle.load(fp)
    return obj


def open_yaml(path: str) -> dict[str, Any] | list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


def json_repr_handler(obj: Any, simple: bool = False) -> Any:
    """! Recursively converts objects into a json representation"""
    attr = "__simplejsonrepr__" if simple else "__jsonrepr__"

    # Recursive: catch end nodes
    if obj is None:
        return None
    if isinstance(obj, int) or isinstance(obj, float) or isinstance(obj, bool) or isinstance(obj, str):
        return obj
    if isinstance(obj, np.int32):  # type: ignore
        return int(obj)  # type: ignore
    if hasattr(obj, attr):
        return obj.__simplejsonrepr__() if simple else obj.__jsonrepr__()

    # Recursive calls
    if isinstance(obj, dict):
        return {json_repr_handler(k, simple): json_repr_handler(v, simple) for k, v in obj.items()}  # type: ignore
    if isinstance(obj, set):
        return json_repr_handler(list(obj), simple)  # type: ignore
    if isinstance(obj, list):
        return [json_repr_handler(x, simple) for x in obj]  # type: ignore
    if isinstance(obj, tuple):
        return tuple(json_repr_handler(x, simple) for x in obj)  # type: ignore

    raise TypeError(f"Object of type {type(obj)} is not serializable. Create a {attr} method.")


class UniqueMessageFilter(logging.Filter):
    """! Prevents the logger from filtering duplicate messages"""

    def __init__(self):
        super().__init__()
        self.recorded_messages: set[str] = set()

    def filter(self, record: logging.LogRecord):
        message = record.getMessage()
        if message in self.recorded_messages:
            return False  # Skip this message
        else:
            self.recorded_messages.add(message)
            return True


T = TypeVar("T")


@no_type_check
class DiGraphWrapper(Generic[T], DiGraph):
    """Wraps the DiGraph class with type annotations for the nodes"""

    @overload
    def in_edges(self, node: T, data: Literal[False]) -> list[tuple[T, T]]: ...

    @overload
    def in_edges(self, node: T, data: Literal[True]) -> list[tuple[T, T, dict[str, Any]]]: ...

    @overload
    def in_edges(self, node: T) -> list[tuple[T, T]]: ...

    def in_edges(  # type: ignore # pylint: disable=W0246
        self,
        node: T,
        data: bool = False,
    ) -> list[tuple[T, T]] | list[tuple[T, T, dict[str, Any]]]:
        return super().in_edges(node, data)  # type: ignore

    @overload
    def out_edges(self, node: T, data: Literal[True]) -> list[tuple[T, T, dict[str, Any]]]: ...

    @overload
    def out_edges(self, node: T, data: Literal[False]) -> list[tuple[T, T]]: ...

    @overload
    def out_edges(self, node: T) -> list[tuple[T, T]]: ...

    def out_edges(  # type: ignore # pylint: disable=W0246
        self,
        node: T,
        data: bool = False,
    ) -> list[tuple[T, T]] | list[tuple[T, T, dict[str, Any]]]:
        return super().out_edges(node, data)  # type: ignore

    @typeguard_ignore
    def in_degree(self) -> Iterator[tuple[T, int]]:  # type: ignore
        return super().in_degree()  # type: ignore

    @overload
    def out_degree(self, node: Literal[None]) -> Iterator[tuple[T, int]]: ...

    @overload
    def out_degree(self) -> Iterator[tuple[T, int]]: ...

    @overload
    def out_degree(self, node: T) -> int: ...

    def out_degree(self, node: T | None = None) -> int | Iterator[tuple[T, int]]:  # type: ignore
        if node:
            return super().out_degree(node)  # type: ignore
        return super().out_degree()  # type: ignore

    def successors(self, node: T) -> Iterator[T]:  # type: ignore # pylint: disable=W0246
        return super().successors(node)  # type: ignore

    def predecessors(self, node: T) -> Iterator[T]:  # type: ignore # pylint: disable=W0246
        return super().predecessors(node)  # type: ignore

    @typeguard_ignore
    def topological_sort(self) -> Iterator[T]:
        return nx.topological_sort(self)  # type: ignore

    def add_node(self, node: T) -> None:  # type: ignore # pylint: disable=W0246
        super().add_node(node)  # type: ignore

    def add_nodes_from(self, node: Sequence[T]) -> None:  # pylint: disable=W0246
        super().add_nodes_from(node)  # type: ignore

    def remove_nodes_from(self, nodes: Iterator[T]) -> None:  # pylint: disable=W0246
        super().remove_nodes_from(nodes)  # type: ignore

    def add_edge(self, edge_from: T, edge_to: T) -> None:  # type: ignore # pylint: disable=W0246
        super().add_edge(edge_from, edge_to)  # type: ignore

    def add_edges_from(  # type: ignore # pylint: disable=W0246
        self,
        edges: Sequence[tuple[T, T] | tuple[T, T, Any]],
    ) -> None:
        super().add_edges_from(edges)  # type: ignore

    def all_simple_paths(self, producer: T, consumer: T) -> Iterator[list[T]]:
        return nx.all_simple_paths(self, source=producer, target=consumer)  # type: ignore

    def shortest_path(self, producer: T, consumer: T) -> list[T]:
        return nx.shortest_path(self, producer, consumer)  # type: ignore

    @property
    def node_list(self) -> list[T]:
        return list(self.nodes())  # type: ignore

    def get_node_with_id(self, node_id: int) -> T:
        for node in self.node_list:
            if node.id == node_id:  # type: ignore
                return node
        raise ValueError(f"Node with id {node_id} not found.")
