from abc import ABCMeta, abstractmethod
from typing import Any

from onnx import ModelProto, NodeProto

from zigzag.workload.DummyNode import DummyNode
from zigzag.workload.layer_node import LayerNode


class Parser(metaclass=ABCMeta):
    """! Abstract base class that represents a parser of an onnx operator. Example: Conv, MatMul, etc."""

    def __init__(
        self,
        node_id: int,
        node: NodeProto,
        nodes_outputs: dict[int, Any],
        mapping: dict[str, dict[str, Any]] | None,
        onnx_model: ModelProto | None,
    ) -> None:
        self.node_id = node_id
        self.node = node
        self.nodes_outputs = nodes_outputs
        self.mapping = mapping
        self.onnx_model = onnx_model

    @abstractmethod
    def run(self) -> LayerNode | DummyNode: ...
