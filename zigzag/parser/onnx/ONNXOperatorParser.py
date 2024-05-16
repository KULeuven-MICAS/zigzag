from abc import ABCMeta, abstractmethod
from typing import Any

from onnx import ModelProto, NodeProto

from zigzag.parser.onnx.utils import get_onnx_tensor_type
from zigzag.workload.LayerNodeABC import LayerNodeABC


class ONNXOperatorParser(metaclass=ABCMeta):
    """! Abstract base class that represents a parser of an onnx operator. Example: Conv, MatMul, etc."""

    def __init__(
        self,
        node_id: int,
        node: NodeProto,
        nodes_outputs: dict[int, Any],
        onnx_model: ModelProto,
    ) -> None:
        self.node_id = node_id
        self.node_name = node.name if node.name else f"Layer{node_id}"
        self.node = node
        self.nodes_outputs = nodes_outputs
        self.onnx_model = onnx_model

    @abstractmethod
    def run(self) -> LayerNodeABC: ...

    def get_input_output_weight_data_type(self):
        """! Return the data type of the input, output and weight tensors of this node."""
        input_name = self.node.input[0]
        output_name = self.node.output[0]
        weight_name = self.get_weight_name(self.node)

        input_elem_type = get_onnx_tensor_type(input_name, self.onnx_model).elem_type
        output_elem_type = get_onnx_tensor_type(output_name, self.onnx_model).elem_type
        weight_elem_type = get_onnx_tensor_type(weight_name, self.onnx_model).elem_type

        return input_elem_type, output_elem_type, weight_elem_type

    def get_weight_name(self, node: NodeProto):
        """! Return the name of the weight input of this node depending on its operator type
        @param node (NodeProto): The node
        """
        op_type = node.op_type  # 'Conv', 'QLinearConv', ...
        if op_type == "Conv":
            return node.input[1]
        elif op_type == "QLinearConv":
            return node.input[3]
        else:
            raise NotImplementedError(f"Retrieving weight name for onnx node of type {op_type} is not supported.")
