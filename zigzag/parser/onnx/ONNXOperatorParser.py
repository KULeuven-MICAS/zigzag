from abc import ABCMeta, abstractmethod
from typing import Any

from onnx import NodeProto

from zigzag.workload.DummyNode import DummyNode
from zigzag.workload.layer_node import LayerNode


class ONNXOperatorParser(metaclass=ABCMeta):
    """! Abstract base class that represents a parser of an onnx operator. Example: Conv, MatMul, etc."""

    def __init__(
        self,
        node_id: int,
        node: NodeProto,
        nodes_outputs: dict[int, Any],
    ) -> None:
        self.node_id = node_id
        self.node = node
        self.nodes_outputs = nodes_outputs

    @abstractmethod
    def run(self) -> LayerNode | DummyNode: ...
